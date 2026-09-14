#include "MediaPipeLiveLinkSource.h"
#include "ILiveLinkClient.h"
#include "LiveLinkTypes.h"
#include "Roles/LiveLinkAnimationRole.h"
#include "OSCManager.h"

const FName FMediaPipeLiveLinkSource::DefaultSubjectName(TEXT("MediaPipePose"));

// Indices 0-21: Manny's body, unchanged. Indices 22-59: 19 finger bones per
// hand (a metacarpal + 3 phalanges for index/middle/ring/pinky, 3 phalanges
// only for the thumb - Manny has no thumb metacarpal), appended by
// hand_solver.py after pose_solver.py's 22 body bones - see that file's
// FINGER_CHAIN for where these names/parents come from (a RefSkeleton dump
// of SKM_Manny_Simple, not guessed).
static const TArray<FName> MediaPipeBoneNames = {
    TEXT("pelvis"), TEXT("spine_01"), TEXT("spine_02"), TEXT("spine_04"), TEXT("neck_01"), TEXT("head"),
    TEXT("clavicle_l"), TEXT("upperarm_l"), TEXT("lowerarm_l"), TEXT("hand_l"),
    TEXT("clavicle_r"), TEXT("upperarm_r"), TEXT("lowerarm_r"), TEXT("hand_r"),
    TEXT("thigh_l"), TEXT("calf_l"), TEXT("foot_l"), TEXT("ball_l"),
    TEXT("thigh_r"), TEXT("calf_r"), TEXT("foot_r"), TEXT("ball_r"),

    TEXT("thumb_01_l"), TEXT("thumb_02_l"), TEXT("thumb_03_l"),
    TEXT("index_metacarpal_l"), TEXT("index_01_l"), TEXT("index_02_l"), TEXT("index_03_l"),
    TEXT("middle_metacarpal_l"), TEXT("middle_01_l"), TEXT("middle_02_l"), TEXT("middle_03_l"),
    TEXT("ring_metacarpal_l"), TEXT("ring_01_l"), TEXT("ring_02_l"), TEXT("ring_03_l"),
    TEXT("pinky_metacarpal_l"), TEXT("pinky_01_l"), TEXT("pinky_02_l"), TEXT("pinky_03_l"),

    TEXT("thumb_01_r"), TEXT("thumb_02_r"), TEXT("thumb_03_r"),
    TEXT("index_metacarpal_r"), TEXT("index_01_r"), TEXT("index_02_r"), TEXT("index_03_r"),
    TEXT("middle_metacarpal_r"), TEXT("middle_01_r"), TEXT("middle_02_r"), TEXT("middle_03_r"),
    TEXT("ring_metacarpal_r"), TEXT("ring_01_r"), TEXT("ring_02_r"), TEXT("ring_03_r"),
    TEXT("pinky_metacarpal_r"), TEXT("pinky_01_r"), TEXT("pinky_02_r"), TEXT("pinky_03_r"),
};

static const TArray<int32> MediaPipeBoneParents = {
    -1, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20,

    // left hand (parent 9 = hand_l)
    9, 22, 23,
    9, 25, 26, 27,
    9, 29, 30, 31,
    9, 33, 34, 35,
    9, 37, 38, 39,

    // right hand (parent 13 = hand_r)
    13, 41, 42,
    13, 44, 45, 46,
    13, 48, 49, 50,
    13, 52, 53, 54,
    13, 56, 57, 58,
};

FMediaPipeLiveLinkSource::FMediaPipeLiveLinkSource(const FMediaPipeLiveLinkSettings& InSettings)
    : Client(nullptr), Settings(InSettings), OSCServer(nullptr)
{
}

FMediaPipeLiveLinkSource::~FMediaPipeLiveLinkSource()
{
    if (OSCServer)
    {
        OSCServer->Stop();
        OSCServer = nullptr;
    }
}

void FMediaPipeLiveLinkSource::ReceiveClient(ILiveLinkClient* InClient, FGuid InSourceGuid)
{
    Client = InClient;
    SourceGuid = InSourceGuid;

    OSCServer = NewObject<UOSCServer>();
    OSCServer->SetAddress(Settings.LocalEndpoint.Address.ToString(), Settings.LocalEndpoint.Port);

    // Bind the OSC receive event directly to our parse function
    OSCServer->OnOscMessageReceivedNative.AddRaw(this, &FMediaPipeLiveLinkSource::OnOSCMessageReceived);
    OSCServer->Listen();
}

bool FMediaPipeLiveLinkSource::RequestSourceShutdown()
{
    if (OSCServer) OSCServer->Stop();
    return true;
}

FText FMediaPipeLiveLinkSource::GetSourceStatus() const
{
    return OSCServer && OSCServer->IsActive() ? FText::FromString("Listening (OSC)") : FText::FromString("Disconnected");
}

void FMediaPipeLiveLinkSource::AddReferencedObjects(FReferenceCollector& Collector)
{
    if (OSCServer) Collector.AddReferencedObject(OSCServer);
}

void FMediaPipeLiveLinkSource::CreateSubject(const FName& InSubjectName)
{
    FScopeLock Lock(&SubjectsCriticalSection);
    if (!Client || RegisteredSubjects.Contains(InSubjectName)) return;

    RegisteredSubjects.Add(InSubjectName);

    FLiveLinkStaticDataStruct StaticDataStruct(FLiveLinkSkeletonStaticData::StaticStruct());
    FLiveLinkSkeletonStaticData& SkeletonData = *StaticDataStruct.Cast<FLiveLinkSkeletonStaticData>();

    SkeletonData.BoneNames = MediaPipeBoneNames;
    SkeletonData.BoneParents = MediaPipeBoneParents;
    SkeletonData.PropertyNames.Add(FName(TEXT("present")));

    Client->PushSubjectStaticData_AnyThread({ SourceGuid, InSubjectName }, ULiveLinkAnimationRole::StaticClass(), MoveTemp(StaticDataStruct));
}

void FMediaPipeLiveLinkSource::OnOSCMessageReceived(const FOSCMessage& Message, const FString& IPAddress, uint16 Port)
{
    if (!Client || Message.GetAddress().GetFullPath() != TEXT("/mediapipe/pose")) return;

    CreateSubject(DefaultSubjectName);

    TArray<float> Args;
    UOSCManager::GetAllFloats(Message, Args);

    // 1 present flag + 7 floats per bone (3 pos, 4 rot)
    const int32 ExpectedArgs = 1 + (MediaPipeBoneNames.Num() * 7);
    if (Args.Num() < ExpectedArgs) return;

    FLiveLinkFrameDataStruct FrameDataStruct(FLiveLinkAnimationFrameData::StaticStruct());
    FLiveLinkAnimationFrameData& FrameData = *FrameDataStruct.Cast<FLiveLinkAnimationFrameData>();
    FrameData.WorldTime = FLiveLinkWorldTime(FPlatformTime::Seconds());

    FrameData.PropertyValues.Add(Args[0]); // Present flag

    FrameData.Transforms.Reserve(MediaPipeBoneNames.Num());
    int32 ArgIndex = 1;

    for (int32 i = 0; i < MediaPipeBoneNames.Num(); i++)
    {
        FVector Pos(Args[ArgIndex], Args[ArgIndex + 1], Args[ArgIndex + 2]);
        FQuat Rot(Args[ArgIndex + 3], Args[ArgIndex + 4], Args[ArgIndex + 5], Args[ArgIndex + 6]);
        Rot.Normalize();

        // Log every bone every frame
        /*
        UE_LOG(LogTemp, Warning, TEXT("Bone %d (%s): Pos(X=%.1f, Y=%.1f, Z=%.1f) | Rot(X=%.3f, Y=%.3f, Z=%.3f, W=%.3f)"),
            i, *MediaPipeBoneNames[i].ToString(), Pos.X, Pos.Y, Pos.Z, Rot.X, Rot.Y, Rot.Z, Rot.W);
        */
        FrameData.Transforms.Add(FTransform(Rot, Pos));
        ArgIndex += 7;
    }

    Client->PushSubjectFrameData_AnyThread({ SourceGuid, DefaultSubjectName }, MoveTemp(FrameDataStruct));
}
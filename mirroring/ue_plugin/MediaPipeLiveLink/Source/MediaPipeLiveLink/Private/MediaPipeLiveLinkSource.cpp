#include "MediaPipeLiveLinkSource.h"
#include "ILiveLinkClient.h"
#include "LiveLinkTypes.h"
#include "Roles/LiveLinkAnimationRole.h"
#include "OSCManager.h"

const FName FMediaPipeLiveLinkSource::DefaultSubjectName(TEXT("MediaPipePose"));

static const TArray<FName> MediaPipeBoneNames = {
    TEXT("pelvis"), TEXT("spine_01"), TEXT("spine_02"), TEXT("spine_04"), TEXT("neck_01"), TEXT("head"),
    TEXT("clavicle_l"), TEXT("upperarm_l"), TEXT("lowerarm_l"), TEXT("hand_l"),
    TEXT("clavicle_r"), TEXT("upperarm_r"), TEXT("lowerarm_r"), TEXT("hand_r"),
    TEXT("thigh_l"), TEXT("calf_l"), TEXT("foot_l"), TEXT("ball_l"),
    TEXT("thigh_r"), TEXT("calf_r"), TEXT("foot_r"), TEXT("ball_r")
};

static const TArray<int32> MediaPipeBoneParents = {
    -1, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20
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
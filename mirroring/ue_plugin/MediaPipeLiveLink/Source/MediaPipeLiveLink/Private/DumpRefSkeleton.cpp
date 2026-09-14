// Fill out your copyright notice in the Description page of Project Settings.

#include "DumpRefSkeleton.h"

// 1. Removed UFUNCTION macro
// 2. Removed 'static' keyword (only used in the header)
// 3. Added UDumpRefSkeleton:: scope resolution
void UDumpRefSkeleton::DumpRefSkeleton(USkeletalMesh* Mesh)
{
    if (!Mesh) return;
    const FReferenceSkeleton& Ref = Mesh->GetRefSkeleton();
    const TArray<FTransform>& Pose = Ref.GetRefBonePose();
    for (int32 i = 0; i < Ref.GetNum(); ++i)
    {
        const FTransform& T = Pose[i];
        const FRotator Rot = T.GetRotation().Rotator();
        const FVector P = T.GetTranslation();
        UE_LOG(LogTemp, Warning,
            TEXT("%s | Parent=%d | Pos=(%.4f, %.4f, %.4f) | Rot(Pitch=%.6f, Yaw=%.6f, Roll=%.6f)"),
            *Ref.GetBoneName(i).ToString(), Ref.GetParentIndex(i), P.X, P.Y, P.Z,
            Rot.Pitch, Rot.Yaw, Rot.Roll);
    }
}
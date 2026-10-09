// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "Engine/SkeletalMesh.h" 
#include "DumpRefSkeleton.generated.h"

/**
 * 
 */
UCLASS()
class MEDIAPIPELIVELINK_API UDumpRefSkeleton : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public: // Must be public for Blueprints to see it

	UFUNCTION(BlueprintCallable, Category = "MediaPipe Live Link: Custom Debug")
	static void DumpRefSkeleton(USkeletalMesh* Mesh); // Now safely inside the class
	
};
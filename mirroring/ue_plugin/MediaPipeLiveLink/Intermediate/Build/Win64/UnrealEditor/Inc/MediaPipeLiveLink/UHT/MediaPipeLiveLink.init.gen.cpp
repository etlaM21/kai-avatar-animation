// Copyright Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeMediaPipeLiveLink_init() {}
static_assert(!UE_WITH_CONSTINIT_UOBJECT, "This generated code can only be compiled with !UE_WITH_CONSTINIT_OBJECT");
	static FPackageRegistrationInfo Z_Registration_Info_UPackage__Script_MediaPipeLiveLink;
	FORCENOINLINE UPackage* Z_Construct_UPackage__Script_MediaPipeLiveLink(ETypeConstructPhase)
	{
		if (!Z_Registration_Info_UPackage__Script_MediaPipeLiveLink.OuterSingleton)
		{
		static const UECodeGen_Private::FPackageParams PackageParams = {
			"/Script/MediaPipeLiveLink",
			nullptr,
			0,
			PKG_CompiledIn | 0x00000000,
			0xF6B8B818,
			0x11F36C6E,
			METADATA_PARAMS(0, nullptr)
		};
		UECodeGen_Private::ConstructUPackage(Z_Registration_Info_UPackage__Script_MediaPipeLiveLink.OuterSingleton, PackageParams);
	}
	return Z_Registration_Info_UPackage__Script_MediaPipeLiveLink.OuterSingleton;
}
static FRegisterCompiledInInfo Z_CompiledInDeferPackage_UPackage__Script_MediaPipeLiveLink(Z_Construct_UPackage__Script_MediaPipeLiveLink, TEXT("/Script/MediaPipeLiveLink"), Z_Registration_Info_UPackage__Script_MediaPipeLiveLink, CONSTRUCT_RELOAD_VERSION_INFO(FPackageReloadVersionInfo, 0xF6B8B818, 0x11F36C6E));
PRAGMA_ENABLE_DEPRECATION_WARNINGS

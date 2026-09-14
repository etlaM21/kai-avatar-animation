// Copyright Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "MediaPipeLiveLinkSourceFactory.h"

PRAGMA_DISABLE_DEPRECATION_WARNINGS
static_assert(!UE_WITH_CONSTINIT_UOBJECT, "This generated code can only be compiled with !UE_WITH_CONSTINIT_UOBJECT");
void EmptyLinkFunctionForGeneratedCodeMediaPipeLiveLinkSourceFactory() {}

// ********** Begin Cross Module References ********************************************************
LIVELINKINTERFACE_API UClass* Z_Construct_UClass_ULiveLinkSourceFactory(ETypeConstructPhase);
// ********** End Cross Module References **********************************************************

// ********** Begin Same Module References *********************************************************
UPackage* Z_Construct_UPackage__Script_MediaPipeLiveLink(ETypeConstructPhase);
MEDIAPIPELIVELINK_API UClass* Z_Construct_UClass_UMediaPipeLiveLinkSourceFactory(ETypeConstructPhase);
MEDIAPIPELIVELINK_API UClass* Z_Construct_UClass_UMediaPipeLiveLinkSourceFactory(ETypeConstructPhase);
// ********** End Same Module References ***********************************************************
#define UHT_STRUCT_BASE(INIT) UE::CodeGen::ConstInit::TCompiledInObjectPtr<const FStructBaseChain>(UE::Private::AsStructBaseChain(INIT))

// ********** Begin Class UMediaPipeLiveLinkSourceFactory ******************************************
#ifdef UHT_STATICS
#error UHT_STATICS already defined
#endif
#define UHT_STATICS Z_Construct_UClass_UMediaPipeLiveLinkSourceFactory_Statics
struct UHT_STATICS
{
#if WITH_METADATA
	static constexpr UECodeGen_Private::FMetaDataPairParam Type_MetaData[] = {
		{ "IncludePath", "MediaPipeLiveLinkSourceFactory.h" },
		{ "ModuleRelativePath", "Public/MediaPipeLiveLinkSourceFactory.h" },
	};
#endif // WITH_METADATA

// ********** Begin Class UMediaPipeLiveLinkSourceFactory constinit property declarations **********
// ********** End Class UMediaPipeLiveLinkSourceFactory constinit property declarations ************
	static FTypeConstructFunc* DependentSingletons[];
	static constexpr FCppClassTypeInfoStatic StaticCppClassTypeInfo = {
		TCppClassTypeTraits<UMediaPipeLiveLinkSourceFactory>::IsAbstract,
	};
	static const UECodeGen_Private::FClassParams ClassParams;
}; // struct UHT_STATICS
FTypeConstructFunc* UHT_STATICS::DependentSingletons[] = {
	(FTypeConstructFunc*)Z_Construct_UClass_ULiveLinkSourceFactory,
	(FTypeConstructFunc*)Z_Construct_UPackage__Script_MediaPipeLiveLink,
};
static_assert(UE_ARRAY_COUNT(UHT_STATICS::DependentSingletons) < 16);
const UECodeGen_Private::FClassParams UHT_STATICS::ClassParams = {
	&Z_Construct_UClass_UMediaPipeLiveLinkSourceFactory,
	nullptr,
	&StaticCppClassTypeInfo,
	DependentSingletons,
	nullptr,
	nullptr,
	nullptr,
	UE_ARRAY_COUNT(DependentSingletons),
	0,
	0,
	0,
	0x001000A0u,
	METADATA_PARAMS(UE_ARRAY_COUNT(UHT_STATICS::Type_MetaData), UHT_STATICS::Type_MetaData)
};
FClassRegistrationInfo Z_Registration_Info_UClass_UMediaPipeLiveLinkSourceFactory;
UClass* Z_Construct_UClass_UMediaPipeLiveLinkSourceFactory(ETypeConstructPhase Phase)
{
	if (Phase == ETypeConstructPhase::Inner)
	{
		using TClass = UMediaPipeLiveLinkSourceFactory;
		if (!Z_Registration_Info_UClass_UMediaPipeLiveLinkSourceFactory.InnerSingleton)
		{
			GetPrivateStaticClassBody(
				TClass::StaticPackage(),
				TEXT("MediaPipeLiveLinkSourceFactory"),
				Z_Registration_Info_UClass_UMediaPipeLiveLinkSourceFactory.InnerSingleton,
				nullptr,
				DataSizeOf<TClass>(),
				alignof(TClass),
				TClass::StaticClassFlags,
				TClass::StaticClassCastFlags(),
				TClass::StaticConfigName(),
				(UClass::ClassConstructorType)InternalConstructor<TClass>,
				(UClass::ClassVTableHelperCtorCallerType)InternalVTableHelperCtorCaller<TClass>,
				UOBJECT_CPPCLASS_STATICFUNCTIONS_FORCLASS(TClass),
				&TClass::Super::StaticClass,
				&TClass::WithinClass::StaticClass
			);
		}
		return Z_Registration_Info_UClass_UMediaPipeLiveLinkSourceFactory.InnerSingleton;
	}
	if (!Z_Registration_Info_UClass_UMediaPipeLiveLinkSourceFactory.OuterSingleton)
	{
		UECodeGen_Private::ConstructUClass(Z_Registration_Info_UClass_UMediaPipeLiveLinkSourceFactory.OuterSingleton, UHT_STATICS::ClassParams);
	}
	return Z_Registration_Info_UClass_UMediaPipeLiveLinkSourceFactory.OuterSingleton;
}
#undef UHT_STATICS
UMediaPipeLiveLinkSourceFactory::UMediaPipeLiveLinkSourceFactory(const FObjectInitializer& ObjectInitializer) : Super(ObjectInitializer) {}
DEFINE_VTABLE_PTR_HELPER_CTOR_NS(, UMediaPipeLiveLinkSourceFactory);
UMediaPipeLiveLinkSourceFactory::~UMediaPipeLiveLinkSourceFactory() {}
// ********** End Class UMediaPipeLiveLinkSourceFactory ********************************************

// ********** Begin Registration *******************************************************************
#ifdef UHT_STATICS
#error UHT_STATICS already defined
#endif
#define UHT_STATICS Z_CompiledInDeferFile_FID_KaiTracking_Plugins_MediaPipeLiveLink_Source_MediaPipeLiveLink_Public_MediaPipeLiveLinkSourceFactory_h__Script_MediaPipeLiveLink_Statics
struct UHT_STATICS
{
	static constexpr FClassRegisterCompiledInInfo ClassInfo[] = {
		{ Z_Construct_UClass_UMediaPipeLiveLinkSourceFactory, TEXT("UMediaPipeLiveLinkSourceFactory"), &Z_Registration_Info_UClass_UMediaPipeLiveLinkSourceFactory, CONSTRUCT_RELOAD_VERSION_INFO(FClassReloadVersionInfo, sizeof(UMediaPipeLiveLinkSourceFactory), 3586810486U) },
	};
}; // UHT_STATICS 
static FRegisterCompiledInInfo Z_CompiledInDeferFile_FID_KaiTracking_Plugins_MediaPipeLiveLink_Source_MediaPipeLiveLink_Public_MediaPipeLiveLinkSourceFactory_h__Script_MediaPipeLiveLink_efbd4ed4e44a18bf814bba82fb782e831196a87b{
	TEXT("/Script/MediaPipeLiveLink"),
	UHT_STATICS::ClassInfo, UE_ARRAY_COUNT(UHT_STATICS::ClassInfo),
	nullptr, 0,
	nullptr, 0,
	nullptr, 0,
};
#undef UHT_STATICS
// ********** End Registration *********************************************************************
#undef UHT_STRUCT_BASE

PRAGMA_ENABLE_DEPRECATION_WARNINGS

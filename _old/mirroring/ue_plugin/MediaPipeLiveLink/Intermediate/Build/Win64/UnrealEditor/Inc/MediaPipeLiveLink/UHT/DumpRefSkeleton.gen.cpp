// Copyright Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "DumpRefSkeleton.h"

PRAGMA_DISABLE_DEPRECATION_WARNINGS
static_assert(!UE_WITH_CONSTINIT_UOBJECT, "This generated code can only be compiled with !UE_WITH_CONSTINIT_UOBJECT");
void EmptyLinkFunctionForGeneratedCodeDumpRefSkeleton() {}

// ********** Begin Cross Module References ********************************************************
ENGINE_API UClass* Z_Construct_UClass_UBlueprintFunctionLibrary(ETypeConstructPhase);
ENGINE_API UClass* Z_Construct_UClass_USkeletalMesh(ETypeConstructPhase);
// ********** End Cross Module References **********************************************************

// ********** Begin Same Module References *********************************************************
UPackage* Z_Construct_UPackage__Script_MediaPipeLiveLink(ETypeConstructPhase);
MEDIAPIPELIVELINK_API UClass* Z_Construct_UClass_UDumpRefSkeleton(ETypeConstructPhase);
MEDIAPIPELIVELINK_API UClass* Z_Construct_UClass_UDumpRefSkeleton(ETypeConstructPhase);
// ********** End Same Module References ***********************************************************
#define UHT_STRUCT_BASE(INIT) UE::CodeGen::ConstInit::TCompiledInObjectPtr<const FStructBaseChain>(UE::Private::AsStructBaseChain(INIT))

// ********** Begin Class UDumpRefSkeleton Function DumpRefSkeleton ********************************
#ifdef UHT_STATICS
#error UHT_STATICS already defined
#endif
#define UHT_STATICS Z_Construct_UFunction_UDumpRefSkeleton_DumpRefSkeleton_Statics
struct UHT_STATICS
{
	struct DumpRefSkeleton_eventDumpRefSkeleton_Parms
	{
		USkeletalMesh* Mesh;
	};
#if WITH_METADATA
	static constexpr UECodeGen_Private::FMetaDataPairParam Type_MetaData[] = {
		{ "Category", "MediaPipe Live Link: Custom Debug" },
#if !UE_BUILD_SHIPPING
		{ "Comment", "// Must be public for Blueprints to see it\n" },
#endif
		{ "ModuleRelativePath", "Public/DumpRefSkeleton.h" },
#if !UE_BUILD_SHIPPING
		{ "ToolTip", "Must be public for Blueprints to see it" },
#endif
	};
#endif // WITH_METADATA

// ********** Begin Function DumpRefSkeleton constinit property declarations ***********************
	static const UECodeGen_Private::FObjectPropertyParams NewProp_Mesh;
	static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
// ********** End Function DumpRefSkeleton constinit property declarations *************************
	static const UECodeGen_Private::FFunctionParams FuncParams;
};

// ********** Begin Function DumpRefSkeleton Property Definitions **********************************
const UECodeGen_Private::FObjectPropertyParams UHT_STATICS::NewProp_Mesh = { "Mesh", nullptr, (EPropertyFlags)0x0010000000000080, UECodeGen_Private::EPropertyGenFlags::Object, nullptr, nullptr, 1, STRUCT_OFFSET(DumpRefSkeleton_eventDumpRefSkeleton_Parms, Mesh), Z_Construct_UClass_USkeletalMesh, METADATA_PARAMS(0, nullptr) };
const UECodeGen_Private::FPropertyParamsBase* const UHT_STATICS::PropPointers[] = {
	(const UECodeGen_Private::FPropertyParamsBase*)&UHT_STATICS::NewProp_Mesh,
};
static_assert(UE_ARRAY_COUNT(UHT_STATICS::PropPointers) < 2048);
// ********** End Function DumpRefSkeleton Property Definitions ************************************
const UECodeGen_Private::FFunctionParams UHT_STATICS::FuncParams = { { (FTypeConstructFunc*)Z_Construct_UClass_UDumpRefSkeleton, nullptr, "DumpRefSkeleton", UHT_STATICS::PropPointers, UE_ARRAY_COUNT(UHT_STATICS::PropPointers), DataSizeOf<UHT_STATICS::DumpRefSkeleton_eventDumpRefSkeleton_Parms>(), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04022401, 0, 0, METADATA_PARAMS(UE_ARRAY_COUNT(UHT_STATICS::Type_MetaData), UHT_STATICS::Type_MetaData)},  };
static_assert(sizeof(UHT_STATICS::DumpRefSkeleton_eventDumpRefSkeleton_Parms) < MAX_uint16);
UFunction* Z_Construct_UFunction_UDumpRefSkeleton_DumpRefSkeleton(ETypeConstructPhase Phase)
{
	static UFunction* ReturnFunction = nullptr;
	if (!ReturnFunction)
	{
		UECodeGen_Private::ConstructUFunction(&ReturnFunction, UHT_STATICS::FuncParams);
	}
	return ReturnFunction;
}
#undef UHT_STATICS
DEFINE_FUNCTION(UDumpRefSkeleton::execDumpRefSkeleton)
{
	P_GET_OBJECT(USkeletalMesh,Z_Param_Mesh);
	P_FINISH;
	P_NATIVE_BEGIN;
	UDumpRefSkeleton::DumpRefSkeleton(Z_Param_Mesh);
	P_NATIVE_END;
}
// ********** End Class UDumpRefSkeleton Function DumpRefSkeleton **********************************

// ********** Begin Class UDumpRefSkeleton *********************************************************
#ifdef UHT_STATICS
#error UHT_STATICS already defined
#endif
#define UHT_STATICS Z_Construct_UClass_UDumpRefSkeleton_Statics
struct UHT_STATICS
{
#if WITH_METADATA
	static constexpr UECodeGen_Private::FMetaDataPairParam Type_MetaData[] = {
#if !UE_BUILD_SHIPPING
		{ "Comment", "/**\n * \n */" },
#endif
		{ "IncludePath", "DumpRefSkeleton.h" },
		{ "ModuleRelativePath", "Public/DumpRefSkeleton.h" },
	};
#endif // WITH_METADATA

// ********** Begin Class UDumpRefSkeleton constinit property declarations *************************
// ********** End Class UDumpRefSkeleton constinit property declarations ***************************
	static constexpr UE::CodeGen::FClassNativeFunction Funcs[] = {
		{ .NameUTF8 = UTF8TEXT("DumpRefSkeleton"), .Pointer = &UDumpRefSkeleton::execDumpRefSkeleton },
	};
	static FTypeConstructFunc* DependentSingletons[];
	static constexpr FClassFunctionLinkInfo FuncInfo[] = {
		{ &Z_Construct_UFunction_UDumpRefSkeleton_DumpRefSkeleton, "DumpRefSkeleton" }, // 9493b4bd50755e7c49492b4dcf23aae75c43d86d
	};
	static_assert(UE_ARRAY_COUNT(FuncInfo) < 2048);
	static constexpr FCppClassTypeInfoStatic StaticCppClassTypeInfo = {
		TCppClassTypeTraits<UDumpRefSkeleton>::IsAbstract,
	};
	static const UECodeGen_Private::FClassParams ClassParams;
}; // struct UHT_STATICS
FTypeConstructFunc* UHT_STATICS::DependentSingletons[] = {
	(FTypeConstructFunc*)Z_Construct_UClass_UBlueprintFunctionLibrary,
	(FTypeConstructFunc*)Z_Construct_UPackage__Script_MediaPipeLiveLink,
};
static_assert(UE_ARRAY_COUNT(UHT_STATICS::DependentSingletons) < 16);
const UECodeGen_Private::FClassParams UHT_STATICS::ClassParams = {
	&Z_Construct_UClass_UDumpRefSkeleton,
	nullptr,
	&StaticCppClassTypeInfo,
	DependentSingletons,
	FuncInfo,
	nullptr,
	nullptr,
	UE_ARRAY_COUNT(DependentSingletons),
	UE_ARRAY_COUNT(FuncInfo),
	0,
	0,
	0x001000A0u,
	METADATA_PARAMS(UE_ARRAY_COUNT(UHT_STATICS::Type_MetaData), UHT_STATICS::Type_MetaData)
};
static void UDumpRefSkeleton_StaticRegisterNativesUDumpRefSkeleton()
{
	UClass* Class = UDumpRefSkeleton::StaticClass();
	FNativeFunctionRegistrar::RegisterFunctions(Class, 		MakeConstArrayView(UHT_STATICS::Funcs));
}
FClassRegistrationInfo Z_Registration_Info_UClass_UDumpRefSkeleton;
UClass* Z_Construct_UClass_UDumpRefSkeleton(ETypeConstructPhase Phase)
{
	if (Phase == ETypeConstructPhase::Inner)
	{
		using TClass = UDumpRefSkeleton;
		if (!Z_Registration_Info_UClass_UDumpRefSkeleton.InnerSingleton)
		{
			GetPrivateStaticClassBody(
				TClass::StaticPackage(),
				TEXT("DumpRefSkeleton"),
				Z_Registration_Info_UClass_UDumpRefSkeleton.InnerSingleton,
				UDumpRefSkeleton_StaticRegisterNativesUDumpRefSkeleton,
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
		return Z_Registration_Info_UClass_UDumpRefSkeleton.InnerSingleton;
	}
	if (!Z_Registration_Info_UClass_UDumpRefSkeleton.OuterSingleton)
	{
		UECodeGen_Private::ConstructUClass(Z_Registration_Info_UClass_UDumpRefSkeleton.OuterSingleton, UHT_STATICS::ClassParams);
	}
	return Z_Registration_Info_UClass_UDumpRefSkeleton.OuterSingleton;
}
#undef UHT_STATICS
UDumpRefSkeleton::UDumpRefSkeleton(const FObjectInitializer& ObjectInitializer) : Super(ObjectInitializer) {}
DEFINE_VTABLE_PTR_HELPER_CTOR_NS(, UDumpRefSkeleton);
UDumpRefSkeleton::~UDumpRefSkeleton() {}
// ********** End Class UDumpRefSkeleton ***********************************************************

// ********** Begin Registration *******************************************************************
#ifdef UHT_STATICS
#error UHT_STATICS already defined
#endif
#define UHT_STATICS Z_CompiledInDeferFile_FID_KaiTracking_Plugins_MediaPipeLiveLink_Source_MediaPipeLiveLink_Public_DumpRefSkeleton_h__Script_MediaPipeLiveLink_Statics
struct UHT_STATICS
{
	static constexpr FClassRegisterCompiledInInfo ClassInfo[] = {
		{ Z_Construct_UClass_UDumpRefSkeleton, TEXT("UDumpRefSkeleton"), &Z_Registration_Info_UClass_UDumpRefSkeleton, CONSTRUCT_RELOAD_VERSION_INFO(FClassReloadVersionInfo, sizeof(UDumpRefSkeleton), 3610813057U) },
	};
}; // UHT_STATICS 
static FRegisterCompiledInInfo Z_CompiledInDeferFile_FID_KaiTracking_Plugins_MediaPipeLiveLink_Source_MediaPipeLiveLink_Public_DumpRefSkeleton_h__Script_MediaPipeLiveLink_9d1471a7eba9e468699e7a1bfc4d766cfee43afc{
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

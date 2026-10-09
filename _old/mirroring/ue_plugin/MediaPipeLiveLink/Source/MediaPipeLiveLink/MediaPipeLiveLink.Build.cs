using UnrealBuildTool;

public class MediaPipeLiveLink : ModuleRules
{
	public MediaPipeLiveLink(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"LiveLinkInterface",
				"LiveLink",
				"LiveLinkAnimationCore",
				"OSC"
			}
		);

        PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"CoreUObject",
				"Engine",
				"Slate",
				"SlateCore",
				"Sockets",
				"Networking",
				"InputCore",
			}
		);
	}
}
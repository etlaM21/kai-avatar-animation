#include "MediaPipeLiveLinkSourceFactory.h"

#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SNumericEntryBox.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/SBoxPanel.h"
#include "Widgets/Text/STextBlock.h"

#define LOCTEXT_NAMESPACE "MediaPipeLiveLinkSourceFactory"

namespace MediaPipeLiveLinkSourceFactoryUI
{
	class SConnectionPanel : public SCompoundWidget
	{
		SLATE_BEGIN_ARGS(SConnectionPanel) {}
		SLATE_END_ARGS()

	public:
		void Construct(const FArguments& InArgs, UMediaPipeLiveLinkSourceFactory::FOnLiveLinkSourceCreated InOnSourceCreated, const UMediaPipeLiveLinkSourceFactory* InFactory)
		{
			OnSourceCreated = InOnSourceCreated;
			Factory = InFactory;

			ChildSlot
			[
				SNew(SVerticalBox)
				+ SVerticalBox::Slot()
				.AutoHeight()
				.Padding(4)
				[
					SNew(SHorizontalBox)
					+ SHorizontalBox::Slot()
					.HAlign(HAlign_Left)
					.FillWidth(0.5f)
					[
						SNew(STextBlock)
						.Text(LOCTEXT("PortLabel", "UDP Port:"))
					]
					+ SHorizontalBox::Slot()
					.HAlign(HAlign_Fill)
					.FillWidth(0.5f)
					[
						SNew(SNumericEntryBox<uint16>)
						.AllowSpin(true)
						.MinValue(1)
						.MaxValue(65535)
						.Value(this, &SConnectionPanel::GetPortValue)
						.OnValueChanged(this, &SConnectionPanel::OnPortValueChanged)
					]
				]
				+ SVerticalBox::Slot()
				.AutoHeight()
				.Padding(4)
				.HAlign(HAlign_Right)
				[
					SNew(SButton)
					.OnClicked(this, &SConnectionPanel::OnCreateClicked)
					[
						SNew(STextBlock)
						.Text(LOCTEXT("CreateButton", "Create Receiver"))
					]
				]
			];
		}

	private:
		TOptional<uint16> GetPortValue() const
		{
			return Settings.LocalEndpoint.Port;
		}

		void OnPortValueChanged(uint16 NewValue)
		{
			Settings.LocalEndpoint.Port = NewValue;
		}

		FReply OnCreateClicked()
		{
			if (Factory)
			{
				TSharedPtr<ILiveLinkSource> NewSource = MakeShared<FMediaPipeLiveLinkSource>(Settings);
				OnSourceCreated.ExecuteIfBound(NewSource, FString::Printf(TEXT(":%d"), Settings.LocalEndpoint.Port));
			}
			return FReply::Handled();
		}

		FMediaPipeLiveLinkSettings Settings;
		UMediaPipeLiveLinkSourceFactory::FOnLiveLinkSourceCreated OnSourceCreated;
		const UMediaPipeLiveLinkSourceFactory* Factory = nullptr;
	};
}

FText UMediaPipeLiveLinkSourceFactory::GetSourceDisplayName() const
{
	return LOCTEXT("SourceDisplayName", "MediaPipe Pose Live Link");
}

FText UMediaPipeLiveLinkSourceFactory::GetSourceTooltip() const
{
	return LOCTEXT("SourceTooltip", "Receive MediaPipe skeletal pose streaming via UDP JSON");
}

TSharedPtr<SWidget> UMediaPipeLiveLinkSourceFactory::BuildCreationPanel(FOnLiveLinkSourceCreated OnLiveLinkSourceCreated) const
{
	return SNew(MediaPipeLiveLinkSourceFactoryUI::SConnectionPanel, OnLiveLinkSourceCreated, this);
}

TSharedPtr<ILiveLinkSource> UMediaPipeLiveLinkSourceFactory::CreateSource(const FString& ConnectionString) const
{
	FMediaPipeLiveLinkSettings Settings;

	if (!ConnectionString.IsEmpty() && ConnectionString.StartsWith(TEXT(":")))
	{
		uint16 Port = FCString::Atoi(*ConnectionString.Mid(1));
		if (Port > 0)
		{
			Settings.LocalEndpoint.Port = Port;
		}
	}

	return MakeShared<FMediaPipeLiveLinkSource>(Settings);
}

void UMediaPipeLiveLinkSourceFactory::CreateSourceFromSettings(FMediaPipeLiveLinkSettings InSettings, FOnLiveLinkSourceCreated OnLiveLinkSourceCreated) const
{
	TSharedPtr<FMediaPipeLiveLinkSource> NewSource = MakeShared<FMediaPipeLiveLinkSource>(InSettings);
	OnLiveLinkSourceCreated.ExecuteIfBound(NewSource, FString::Printf(TEXT(":%d"), InSettings.LocalEndpoint.Port));
}

#undef LOCTEXT_NAMESPACE
#pragma once

#include "CoreMinimal.h"
#include "LiveLinkSourceFactory.h"
#include "MediaPipeLiveLinkSource.h"
#include "MediaPipeLiveLinkSourceFactory.generated.h"

UCLASS()
class MEDIAPIPELIVELINK_API UMediaPipeLiveLinkSourceFactory : public ULiveLinkSourceFactory
{
	GENERATED_BODY()

public:
	virtual FText GetSourceDisplayName() const override;
	virtual FText GetSourceTooltip() const override;

	virtual EMenuType GetMenuType() const override { return EMenuType::SubPanel; }
	virtual TSharedPtr<SWidget> BuildCreationPanel(FOnLiveLinkSourceCreated OnLiveLinkSourceCreated) const override;
	virtual TSharedPtr<ILiveLinkSource> CreateSource(const FString& ConnectionString) const override;

private:
	void CreateSourceFromSettings(FMediaPipeLiveLinkSettings InSettings, FOnLiveLinkSourceCreated OnLiveLinkSourceCreated) const;
};
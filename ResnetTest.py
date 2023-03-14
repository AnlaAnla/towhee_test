import torch
import torchvision.models as models

model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(in_features=2048, out_features=314)

model.load_state_dict(torch.load(r"D:\Code\ML\model\card_cls\res_card_freeze2.pth"))

print(model)
"""
The template for the students to train the model.
Please do not change the name of the functions in Adv_Training.
"""
import sys
sys.path.append("../../../")
import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils import get_dataset
import importlib.util

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class VirtualModel:
    def __init__(self, device, model) -> None:
        self.device = device
        self.model = model

    def get_batch_output(self, images, with_preprocess=True, skip_detect=False):
        outputs = []
        detect_outputs = []
        for ini_image in images:
            image = torch.unsqueeze(ini_image, 0)
            output = self.model(image).to(self.device)
            outputs.append(output[0])
            detect_outputs.append(0)
        outputs = torch.stack(outputs)
        detect_outputs = torch.tensor(detect_outputs).to(self.device)
        return outputs, detect_outputs

    def get_batch_input_gradient(self, original_images, labels, lossf=None):
        original_images.requires_grad = True
        self.model.eval()
        outputs = self.model(original_images)
        if lossf is None:
            loss = F.nll_loss(outputs, labels)
        else:
            loss = lossf(outputs, labels)
        self.model.zero_grad()
        loss.backward()
        data_grad = original_images.grad.data
        return data_grad


class Adv_Training():
    """
    The class is used to set the defense related to adversarial training and adjust the loss function. Please design your own training methods and add some adversarial examples for training.
    The perturb function is used to generate the adversarial examples for training.
    """
    def __init__(self, device, file_path, target_label=None, epsilon=0.3, min_val=0, max_val=1):
        sys.path.append(file_path)
        from predict import LeNet
        self.model = LeNet().to(device)
        self.epsilon = epsilon
        self.device = device
        self.min_val = min_val
        self.max_val = max_val
        self.target_label = target_label
        self.perturb = self.load_perturb("../attacker_list/nontarget_FGSM")

    def load_perturb(self, attack_path):
        spec = importlib.util.spec_from_file_location('attack', attack_path + '/attack.py')
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        # for attack methods evaluator, the Attack class name should be fixed
        attacker = foo.Attack(VirtualModel(self.device, self.model), self.device, attack_path)
        return attacker


    def train(self, trainset, valset, device, epoches=30):
        self.model.to(device)
        self.model.train()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=30)
        dataset_size = len(trainset)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        for epoch in range(epoches):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                og_inputs=inputs
                og_inputs = og_inputs.to(device)
                og_labels = torch.tensor([1 for i in labels.detach().cpu().tolist()])
                og_labels = og_labels.to(device)

                # zero the parameter gradients
                adv_inputs, _ = self.perturb.attack(og_inputs, labels.detach().cpu().tolist())
                adv_inputs = torch.tensor(adv_inputs).to(device)

                adv_labels = torch.tensor([0 for i in labels.detach().cpu().tolist()])
                adv_labels = adv_labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                og_outputs = self.model(og_inputs)
                loss = criterion(og_outputs, og_labels)
                adv_outputs = self.model(adv_inputs)
                loss += criterion(adv_outputs,adv_labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / dataset_size))
            running_loss = 0.0
        valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=True, num_workers=10)
        correct = 0
        total = 0
        for inputs, labels in valloader:
                #get original inputs and corresponding labels
                og_inputs=inputs
                og_inputs = og_inputs.to(device)
                og_labels = torch.tensor([1 for i in labels.detach().cpu().tolist()])
                og_labels = og_labels.to(device)

                #make predictions on original inputs and compare to corresponding labels
                og_outputs = self.model(og_inputs)
                _, predicted = torch.max(og_outputs.data, 1)
                total += og_labels.size(0)
                correct += (predicted == og_labels).sum().item()

                #get attacked inputs and corresponding labels
                adv_inputs, _ = self.perturb.attack(og_inputs, labels.detach().cpu().tolist())
                adv_inputs = torch.tensor(adv_inputs).to(device)
                adv_labels = torch.tensor([0 for i in labels.detach().cpu().tolist()])
                adv_labels = adv_labels.to(device)

                #make predictions on attacked inputs and compare to corresponding labels
                adv_outputs = self.model(adv_inputs)
                _, predicted = torch.max(adv_outputs.data, 1)
                total += adv_labels.size(0)
                correct += (predicted == adv_labels).sum().item()
                
        print("Accuracy of the detector on the val images: %.3f %%" % (100 * correct / total))
        return


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adv_training = Adv_Training(device, file_path='.')
    dataset_configs = {
                "name": "CIFAR10",
                "binary": True,
                "dataset_path": "../datasets/CIFAR10/student/",
                "student_train_number": 10000,
                "student_val_number": 1000,
                "student_test_number": 100,
    }

    dataset = get_dataset(dataset_configs)
    trainset = dataset['train']
    valset = dataset['val']
    testset = dataset['test']
    adv_training.train(trainset, valset, device)
    torch.save(adv_training.model.state_dict(), "defense_war-detector-model.pth")


if __name__ == "__main__":
    main()
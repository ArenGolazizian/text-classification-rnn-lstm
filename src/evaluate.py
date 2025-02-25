import torch
from tqdm import tqdm
from sklearn.metrics import classification_report

def evaluate_on_test(model, test_loader, device):
    model.eval()
    y_true_test = []
    y_pred_test = []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating on Test Set"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true_test.extend(targets.cpu().numpy())
            y_pred_test.extend(predicted.cpu().numpy())
    print(classification_report(y_true_test, y_pred_test, target_names=['Negative', 'Positive']))

def predict_sentiment(model, input_tensor, device, model_name="Model"):
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        class_label = "Positive" if predicted.item() == 1 else "Negative"
    print(f"The predicted class for the review by {model_name} is: {class_label}")
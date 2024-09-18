import torch
import cv2
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Load YOLOv5 model (e.g., 'yolov5s', 'yolov5m', etc.) 
# pretrained model of yolo
# model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
# model = torch.load('best.pt')['model'].float().fuse().eval()
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt') 

# Load an image using OpenCV
img = cv2.imread(sys.argv[1])

############################################################
# Adjust here own trues and false
# Ploting Graph Based Here
# Example data: Replace with your actual model predictions and ground truth
results_graph = model(sys.argv[1])
predictions = results_graph.pred[0]  # For the first image

# Extract confidence scores and predicted labels
confidence_scores = predictions[:, 4]  # Confidence scores (index 4 in YOLO output)
predicted_labels = predictions[:, 5]  # Predicted class labels

# Combine confidence scores with predictions (if needed)
y_scores = confidence_scores.numpy()

y_trues = []

for x in range(len(y_scores)):
  y_trues.append(x)

# y_scores = [0.1, 0.4, 0.35, 0.8, 0.7, 0.5, 0.9, 0.3, 0.6, 0.2]  # Predicted scores

precision, recall, _ = precision_recall_curve(y_trues, y_scores)

plt.figure()
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('dest/' + sys.argv[1].split("/")[1] +'_precision.png')  # Save the plot to a file
plt.show()
#############################################################


# Convert the image to RGB (YOLOv5 expects images in RGB format)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Perform inference using YOLOv5
results = model(img_rgb)

# Parse results
# The `results` object contains information about the detections
# You can retrieve the detected bounding boxes, labels, and confidences like this:
detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, confidence, class]

# Iterate over detections and draw bounding boxes 
# handling own color and text style here
for x1, y1, x2, y2, conf, cls in detections:
    label = f"{model.names[int(cls)]} {conf:.2f}"  # Class name and confidence
    color = (0, 255, 0)  # Green color for bounding boxes
    
    # Draw rectangle around the object
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    # Put label above the bounding box
    cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)


output_image_path = "dest/" + sys.argv[1].split("/")[1]
cv2.imwrite(output_image_path, img)
# Show the output image
# Want to view directy without saving in buffer mode in frame comment above 34 and 35 line and enable 38, 39, 40
# cv2.imshow("YOLOv5 Output", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

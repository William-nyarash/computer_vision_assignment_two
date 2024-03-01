import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace


def emotions(facial_features):
  return facial_features[0]["dominant_emotion"]

def main():
  image =cv2.imread("johnsample.png")
#   convert the image to gray scale
  grayed_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#    cascade the imega inorder to define the parameters used to capture the face
  classified_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
  face_frame = classified_face.detectMultiScale(grayed_image,scaleFactor=1.2,minNeighbors=3,minSize=(29,29))

  for(tl,tw,bl,bw) in face_frame:
    cv2.rectangle(image,(tl,tw),(tl + bl,tw + bw),(124,76,250),4)
# using deep face analyze the image  emotion
  facial_features = DeepFace.analyze(image,actions=("emotion"))
#   convert the image to RGB format
  marked_face = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  
#   using cv2 built in function add the emotion as the text to the captured frame
  cv2.putText(marked_face,emotions(facial_features),(250,250),cv2.FONT_HERSHEY_SIMPLEX,5,(230,0,120),4,cv2.LINE_AA)

# plot the frame that encloses the face
  plt.figure(figsize=(25,15))
  plt.imshow(marked_face)
  plt.axis('off')
main()

cv2.waitKey(0)
cv2.destroyAllWindows()

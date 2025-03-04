import cv2
import unittest
from preprocess_images import process_image_from_webcam


class TestProcessRealImage(unittest.TestCase):

    def test_process_image(self):
        # Arrange
        test_img_fname = "data/test_snapshot.jpg"
        img = cv2.imread(test_img_fname)

        original_h, original_w, original_c = img.shape

        self.assertEqual(original_c, 3)
        self.assertEqual(original_h, 480)
        self.assertEqual(original_w, 640)

        target_h = 125
        target_w = 125

        # Action
        img = process_image_from_webcam(img, target_size=(target_h,target_w))

        # Assert
        self.assertEqual(len(img.shape), 2)
        actual_h, actual_w = img.shape
        self.assertEqual(actual_h, target_h)
        self.assertEqual(actual_w, target_w)

        # cv2.imshow("processed_img", img)
        # cv2.waitKey(0)



if __name__=="__main__":
    unittest.main()
import cv2
import os
import numpy as np


class ImageStitch:

    """get two image path and stitch them together"""
    def __init__(self, img_path1, img_path2):
        self.img1 = cv2.imread(img_path1)
        self.img2 = cv2.imread(img_path2)
        self.g_img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        self.g_img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        self.final_path = os.path.dirname(img_path1) + r'/merge.jpg'

        self.detect_path = img_path1[:-4] + r'_det.jpg'

        self.sift = cv2.xfeatures2d.SIFT_create()
        self.brute = cv2.BFMatcher()

        self.final_img = None


    def get_keyPoints(self):
        """使用sift计算关键点"""
        kp1, kp2 = {}, {}
        kp1['kp'], kp1['des'] = self.sift.detectAndCompute(self.g_img1, None)
        kp2['kp'], kp2['des'] = self.sift.detectAndCompute(self.g_img2, None)

        return kp1, kp2

    def match(self, kp1, kp2):
        matches = self.brute.knnMatch(kp1['des'], kp2['des'], k=2)

        good_matches = []

        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good_matches.append((m.trainIdx, m.queryIdx))

        if len(good_matches) > 4:
            key_points1 = kp1['kp']
            key_points2 = kp2['kp']

            matched_kp1 = np.float32([key_points1[i].pt for _, i in good_matches])
            matched_kp2 = np.float32([key_points2[i].pt for i, _ in good_matches])

            homo_matrix, _ = cv2.findHomography(matched_kp1, matched_kp2, cv2.RANSAC, 4)

            return homo_matrix
        else:
            return None

    def match_det(self, kp1, kp2):

        """
        DMathc数据结构，包含三个重要的数据：
        queryIdx: 测试图像的特征点描述符的下标， 同时也是描述符对应特征点的下标
        trainIdx: 样本图像的特征点描述符下标，同时也是描述符对应特征点的下标
        distance: 代表这匹配的特征点描述符的欧式距离，数值越小说明两个特征点越相近
        """

        # matches_de = self.brute.match(kp1['des'], kp2['des'])
        # pic = cv2.drawMatches(self.img1, kp1['kp'], self.img2, kp2['kp'], matches_de[:50], None, flags=2)

        matches = self.brute.knnMatch(kp1['des'], kp2['des'], k=2)
        pic = cv2.drawMatchesKnn(self.img1, kp1['kp'], self.img2, kp2['kp'], matches[:200], None, flags=2)
        cv2.imwrite(self.detect_path, pic)
        return self.detect_path

    def image_merge(self, homo_matrix):
        h1, w1 = self.img1.shape[0], self.img2.shape[1]
        h2, w2 = self.img2.shape[0], self.img2.shape[1]

        rect1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape((4, 1, 2))
        rect2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape((4, 1, 2))

        trans_rect = cv2.perspectiveTransform(rect1, homo_matrix)
        total_rect = np.concatenate((rect2, trans_rect), axis=0)
        min_x, min_y = np.int32(total_rect.min(axis=0).ravel())
        max_x, max_y = np.int32(total_rect.max(axis=0).ravel())
        shift_to_zero_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        trans_img = cv2.warpPerspective(self.img1,
                                        shift_to_zero_matrix.dot(homo_matrix),
                                        (max_x - min_x, max_y - min_y))
        trans_img[-min_y:h2 - min_y, -min_x:w2 - min_x] = self.img2

        self.final_img = trans_img
        cv2.imwrite(self.final_path, self.final_img)
        return self.final_path

    def showpic(self):
        cv2.imshow('show', self.final_img)
        if cv2.waitKey() == 27:
            cv2.destroyAllWindows()


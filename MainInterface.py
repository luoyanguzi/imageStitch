from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QGridLayout, QWidget, QPushButton
from PyQt5.QtGui import QDropEvent, QDragEnterEvent, QPixmap
import ImageStitch

class ImageInterface(QMainWindow):
    """拖拽两个图片至主界面，点击对应btn完成相应操作"""

    def __init__(self, parent=None):
        super(ImageInterface, self).__init__(parent)
        self.setMinimumSize(1200, 600)
        self.setAcceptDrops(True)
        self.pix = None
        self.pix_path = None
        self.pic_count = 0

        wig = QWidget(self)
        self.setCentralWidget(wig)
        grid = QGridLayout(wig)

        self.label = QLabel('请输入图', wig), QLabel('请输入图', wig), QLabel('拼接的图片', wig)
        for x in range(3):
            self.label[x].setScaledContents(True)

        grid.addWidget(self.label[0], 0, 0, 3, 3)
        grid.addWidget(self.label[1], 0, 3, 3, 3)
        grid.addWidget(self.label[2], 0, 0, 3, 6)
        self.label[2].hide()

        self.btn = QPushButton('拼接并显示', wig)
        self.btn2 = QPushButton('deleteAll', wig)
        self.btn3 = QPushButton('目标检测', wig)
        grid.addWidget(self.btn, 0, 6, 1, 1)
        grid.addWidget(self.btn2, 2, 6, 1, 1)
        grid.addWidget(self.btn3, 1, 6, 1, 1)  # put above the btn2
        self.btn.clicked.connect(self.on_btn_click)
        self.btn2.clicked.connect(self.on_btn2_click)
        self.btn3.clicked.connect(self.on_btn3_clikc)
        # self.setLayout(grid)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        path = []
        for url in event.mimeData().urls():
            path.append(url.toLocalFile())

        self.processPath(path)

    def on_btn_click(self):
        self.label[0].hide()
        self.label[1].hide()
        self.label[2].show()
        self.stitch()

    def on_btn2_click(self):
        for i in range(2):
            self.label[i].clear()
            self.label[i].show()
            self.label[i].setText('请输入图片')
        self.label[2].hide()
        self.pic_count = 0

    def on_btn3_clikc(self):
        self.label[0].hide()
        self.label[1].hide()
        self.label[2].show()
        self.detect()


    def processPath(self, path) -> None:
        if path is None:
            return None
        if len(path) == 1:
            if self.pic_count == 0:
                self.pix = QPixmap(path[0]), QPixmap(path[0])
                self.pix_path = path[0], path[0]
                self.label[0].setPixmap(self.pix[0])
                self.pic_count += 1
                return None
            if self.pic_count == 1:
                self.pix = self.pix[0], QPixmap(path[0])
                self.pix_path = self.pix_path[0], path[0]
                self.label[1].setPixmap(self.pix[1])
                return None
        if len(path) == 2:
            self.pix = QPixmap(path[0]), QPixmap(path[1])
            self.pix_path = path
            self.label[0].setPixmap(self.pix[0])
            self.label[1].setPixmap(self.pix[1])
            return None

    def stitch(self):
        output = ImageStitch.ImageStitch(self.pix_path[0], self.pix_path[1])
        kp1, kp2 = output.get_keyPoints()
        homo_matrix = output.match(kp1, kp2)
        img_path = output.image_merge(homo_matrix)
        pix = QPixmap(img_path)
        self.label[2].setPixmap(pix)

        # output.showpic()

    def detect(self):
        output = ImageStitch.ImageStitch(self.pix_path[0], self.pix_path[1])
        kp1, kp2 = output.get_keyPoints()
        pic_det = output.match_det(kp1, kp2)
        pix = QPixmap(pic_det)
        self.label[2].setPixmap(pix)



if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    MainWindow = ImageInterface()
    MainWindow.show()

    sys.exit(app.exec())






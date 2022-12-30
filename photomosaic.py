import cv2
import os
import numpy as np
import json
import queue

class PhotoMosaic:
    def __init__(self, step=50, pqsize=20, seed=423) -> None:
        '''
        parameters:
            1. step -> resize the patching images to size (step, step), this parameter also controls the 
        amplification coefficient (step // 10) of the target image. For example, if the step is 100, then 
        the target image will be resized to (10 * row // 100 * 100, 10 * col // 100 * 100), and in the final
        picture, every 100 * 100 square will be replaced by a patching image, so usually the target image
        will be replaced by an image matrix of the size (row // 10, col // 10).
            2. pqsize -> the size of the image pool for each position. The basic technique of the Mosaic
        procedure is to compute the distance between the mean rgb value of a square area of the target image
        and the mean rgb value of all patching images. If we select the closest between the iamges and the 
        square area, the repetition rate wil be very high and the result won't look good. So I applied a 
        Top-K mechanism and use priority queue of pqsize to select the closest pqsize images and then randomly
        select an image among them to fit in the square area.
            3. seed -> used to control reproduction procedures
        '''
        self.original_size = None
        self.step = step            # controls the size of the patching images
        self.coef = step // 10
        self.target = None
        self.imgs = {}              # stores the resized patching images
        self.imgs_data = {}         # stores the mean rgb value of patching images
        self.result = None          # the final photo mosaic result
        self.pqsize = pqsize
        np.random.seed(seed)

    def load_target(self, target_path):
        img = np.array(cv2.imread(target_path))
        self.original_size = img.shape[:2]      # (H, W)
        img = cv2.resize(img, (self.coef * self.original_size[1] // self.step * self.step, self.coef * self.original_size[0] // self.step * self.step))
        self.target = img

    def load_patching_images(self, imgs_path):
        for img_file in os.listdir(imgs_path):
            img = np.array(cv2.imread(os.path.join(imgs_path, img_file)))
            img_unit = cv2.resize(img, (self.step, self.step))
            self.imgs[img_file] = img_unit
            mean, _ = cv2.meanStdDev(img_unit)
            self.imgs_data[img_file] = np.reshape(mean, -1)

    def store_imgs_weight(self, path):
        if len(self.imgs_data) == 0:
            print('No patching images yet, please run the load_patching_images function to get the patching images first.')
            return
        imgs_data = {}
        for k, v in self.imgs_data.items():
            imgs_data[k] = list(v)
        json.dump(imgs_data, open(path, 'w'))

    def process(self):
        if self.target is None:
            print('No target image yet, please run the load_target function to get the target image first.')
            return
        patched_img = np.zeros_like(self.target)
        for i in range(0, self.target.shape[0], self.step):
            print('computing on the {}th row'.format(i))
            for j in range(0, self.target.shape[1], self.step):
                tmp_area = self.target[i:i + self.step, j:j + self.step]    # specify the local square area to replace
                tmp_mean, _ = cv2.meanStdDev(tmp_area)
                tmp_mean = np.reshape(tmp_mean, -1)                         # compute the mean rgb value
                tmp_pq = queue.PriorityQueue()                              # use the priority queue to maintain a list of pqsize patching images

                for img_name in self.imgs.keys():
                    img_mean = self.imgs_data[img_name]
                    tmp_img_dist = np.linalg.norm(tmp_mean - img_mean)      # euclidean distance
                    if tmp_pq.qsize() < self.pqsize:
                        tmp_pq.put((-tmp_img_dist, img_name))
                    else:
                        head = tmp_pq.get()
                        if -head[0] > tmp_img_dist:
                            tmp_pq.put((-tmp_img_dist, img_name))           # replace the image with the largest distance
                        else:
                            tmp_pq.put(head)

                tmp_img_list = []
                while not tmp_pq.empty():
                    head = tmp_pq.get()
                    tmp_img_list.append(head[1])

                idx = np.random.randint(self.pqsize)                        # select a random image among the selected patching images
                patched_img[i:i + self.step, j:j + self.step] = self.imgs[tmp_img_list[idx]]
        
        self.result = patched_img
    
    def store_result(self, path):
        if self.result is None:
            print('No result yet, please run the process function to generate the result image first.')
            return
        cv2.imwrite(path, self.result)

if __name__ == '__main__':
    pqsize = 20
    pm = PhotoMosaic(pqsize=pqsize)
    pm.load_patching_images('monet')
    for i, name in enumerate(os.listdir('xsg')):
        print('processing {}'.format(name))
        pm.load_target(os.path.join('xsg', name))
        pm.process()
        pm.store_result(os.path.join('xsg', '{}-mosaic-{}.jpg'.format(name, i + 1, pqsize)))
        
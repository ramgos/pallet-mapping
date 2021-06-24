import numpy as np
import cv2


# returns the distance squared (taking the root is unnecessary for finding the min distance)
def distance(color1, color2):
    return np.sum((color1 - color2) ** 2, axis=2)


# find out to which color from a set of colors is a color the closest to

def main():
    # load image and pallete

    img = cv2.imread('StarryNight.jpg')
    pallet = cv2.imread('StarryNightPallet.png')

    weight = np.array([1, 1, 1], dtype='float32')  # weight to apply to both pallet and lab
    lab_w = np.array([1, 1, 1], dtype='float32')  # weight to exclusively apply to lab
    pallet_w = np.array([1, 1, 1], dtype='float32')  # weight exclusievly to pallet

    # convert to LAB color mode

    LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    chunk_shape = (100, 100)

    # image needs to be padded so it would fit perfectly the chunk size
    missing_y = 100 - (LAB.shape[0] % chunk_shape[0])
    missing_x = 100 - (LAB.shape[1] % chunk_shape[1])

    LAB = np.pad(LAB, ((0, missing_y), (0, missing_x), (0, 0)))
    LAB_w = np.float32(LAB) * weight * lab_w  # weighted version

    LAB_pallet = cv2.cvtColor(pallet, cv2.COLOR_BGR2LAB)
    LAB_pallet = np.reshape(LAB_pallet, (LAB_pallet.shape[0] * LAB_pallet.shape[1], 3))  # flatten image to 2 dimensional array
    LAB_pallet_w = np.float32(LAB_pallet) * weight * pallet_w  # weighted version

    chunk_y_size = LAB.shape[0]//chunk_shape[0]
    chunk_x_size = LAB.shape[1]//chunk_shape[1]

    # grid of chunks
    chunks = np.zeros((chunk_y_size, chunk_x_size, chunk_shape[0], chunk_shape[1], 3), dtype='uint8')

    for i in range(chunk_y_size):
        for j in range(chunk_x_size):
            # count = (i * chunk_x_size + j) + 1
            # print(f'{count/(chunk_y_size * chunk_x_size) * 100}%')

            # distance of each color in the pallet to each pixel in the image
            distances = np.array([
                distance(
                    LAB_pallet_w[k],
                    LAB_w[chunk_shape[1]*i:chunk_shape[1]*(i+1), chunk_shape[0]*j:chunk_shape[0]*(j+1)]
                ) for k in range(LAB_pallet_w.shape[0])], dtype="uint32"
            )

            # indicies of color from pallet with min distance
            matching_colors = np.argmin(distances, axis=0)
            chunks[i, j] = LAB_pallet[matching_colors]

    # merge chunks
    merged1 = np.concatenate(chunks[::1], axis=1)
    merged2 = np.concatenate(merged1, axis=1)

    # convert to BGR
    BGR = cv2.cvtColor(merged2, cv2.COLOR_LAB2BGR)
    # cut padding
    BGR = BGR[:img.shape[0], :img.shape[1]]
    cv2.imwrite("FinishedResult.png", BGR)


main()
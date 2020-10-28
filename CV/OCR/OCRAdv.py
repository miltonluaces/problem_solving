import os
import cv2
import imutils
import matplotlib.pyplot as plt

# Table identification (only works if there's only one table on a page)
# Params: morph_size, min_text_height_limit, max_text_height_limit, cell_threshold, min_columns


def Preprocess(img, save_in_file, morph_size=(8, 8)):
    pre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #gray scale
    pre = cv2.threshold(pre, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] #otsu thres
    cpy = pre.copy()
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
    cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1) # dilate text
    pre = ~cpy

    if save_in_file is not None:
        cv2.imwrite(save_in_file, pre)
    return pre


def FindTextBoxes(pre, min_text_height_limit=6, max_text_height_limit=40):
    contours = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]

    # Getting the texts bounding boxes based on the text size assumptions
    boxes = []
    for contour in contours:
        box = cv2.boundingRect(contour)
        h = box[3]
        if min_text_height_limit < h < max_text_height_limit:
            boxes.append(box)
    return boxes


def FindTable(boxes, cell_threshold=10, min_columns=2):
    rows = {}
    cols = {}

    # Clustering the bounding boxes by their positions
    for box in boxes:
        (x, y, w, h) = box
        colKey = x // cell_threshold
        rowKey = y // cell_threshold
        cols[rowKey] = [box] if colKey not in cols else cols[colKey] + [box]
        rows[rowKey] = [box] if rowKey not in rows else rows[rowKey] + [box]

    # Filtering out the clusters having less than 2 cols
    tableCells = list(filter(lambda r: len(r) >= min_columns, rows.values()))
    # Sorting the row cells by x coord, then rows by y coord
    tableCells = [list(sorted(tb)) for tb in tableCells]
    tableCells = list(sorted(tableCells, key=lambda r: r[0][1]))

    return tableCells


def BuildLines(tableCells):
    if tableCells is None or len(tableCells) <= 0:
        return [], []

    maxLastColWidthRow = max(tableCells, key=lambda b: b[-1][2])
    maxX = maxLastColWidthRow[-1][0] + maxLastColWidthRow[-1][2]

    maxLastRowHeightBox = max(tableCells[-1], key=lambda b: b[3])
    maxY = maxLastRowHeightBox[1] + maxLastRowHeightBox[3]

    horLines = []
    verLines = []

    for box in tableCells:
        x = box[0][0]
        y = box[0][1]
        horLines.append((x, y, maxX, y))

    for box in tableCells[0]:
        x = box[0]
        y = box[1]
        verLines.append((x, y, x, maxY))

    (x, y, w, h) = tableCells[0][-1]
    verLines.append((maxX, y, maxX, maxY))
    (x, y, w, h) = tableCells[0][0]
    horLines.append((x, maxY, maxX, maxY))

    return horLines, verLines


if __name__ == "__main__":

    # Test

    inFile = os.path.join("../../../../data/Image/" , "inv1.jpeg")
    preFile = os.path.join("../../../../data/Image/" , "inv1pre.jpeg")
    outFile = os.path.join("../../../../data/Image/" , "inv1out.jpeg")
    
    img = cv2.imread("../../../../data/Images/inv1.jpeg")
    plt.imshow(img)
    plt.show()

    preproc = Preprocess(img, preFile)
    textBoxes = FindTextBoxes(preproc)
    tableCells = FindTable(textBoxes)
    horLines, verLines = BuildLines(tableCells)

    # Visualize the result
    vis = img.copy()

    for box in textBoxes:
         (x, y, w, h) = box
         cv2.rectangle(vis, (x, y), (x + w - 2, y + h - 2), (0, 255, 0), 1)

    for line in horLines:
        [x1, y1, x2, y2] = line
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

    for line in verLines:
        [x1, y1, x2, y2] = line
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.imwrite(outFile, vis)
    img = vis
    #img = cv2.imread("../../../../data/Images/inv1out.jpeg")
    plt.imshow(img)
    plt.show()


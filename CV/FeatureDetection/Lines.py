from Utils.Admin.Standard import *
import cv2


# rho : distance resolution in pixels of the Hough grid
# theta : angular resolution in radians of the Hough grid
# thres : minimum number of votes (intersections in Hough grid cell)
# minLineLenght : minimum number of pixels making up a line
# maxLineGap : maximum gap in pixels between connectable line segments 
def GetLines(img, lwrThres, uprThres, rho, theta, thres, minLineLenght, maxLineGap):
    edges = cv2.Canny(img, lwrThres, uprThres)
    lines = cv2.HoughLinesP(edges, rho, theta, thres, np.array([]), minLineLenght, maxLineGap)
    return lines


if __name__=='__main__':
    # Load image
    img = cv2.imread(imgPath + 'charts/chart1.png', 0)
    cv2.imshow("chart", img) 
    cv2.waitKey(0)

    # Get the lines from image
    lines = GetLines(img, lwrThres=50, uprThres=150, rho=1, theta=np.pi/180, thres=15, minLineLenght=50, maxLineGap=20)

    lineImg = np.copy(img) * 0  # creating a blank to draw lines on
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(lineImg,(x1,y1),(x2,y2),(255,0,0),5)

    cv2.imshow("lines", lineImg) 
    cv2.waitKey(0)

    print('Lines :')
    print(lines)

    # Draw the lines on the image
    imgWithLines = cv2.addWeighted(img, 0.8, lineImg, 1, 0)
    cv2.imshow("lines", imgWithLines) 
    cv2.waitKey(0)

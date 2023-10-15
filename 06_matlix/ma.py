import cv2

cap = cv2.VideoCapture(0)
bgCap = cv2.VideoCapture('./bg/bg.mp4')

capSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) # 영상의 너비, 높이
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), capSize)

sub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=100, detectShadows=False)

while cap.isOpened():
    ret, fgImg = cap.read()

    if not ret:
        break

    bgRet, bgImg = bgCap.read()

    if not bgRet:
        bgCap.set(1, 0)
        _, bgImg = bgCap.read()

    bgImg = cv2.resize(bgImg, dsize=capSize)

    mask = sub.apply(fgImg)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    result = cv2.bitwise_and(bgImg, fgImg, mask=mask)

    cv2.imshow('result', result)
    out.write(result)

    if cv2.waitKey(1) == ord('q'):
        break


out.release()
cap.release()
bgCap.release()
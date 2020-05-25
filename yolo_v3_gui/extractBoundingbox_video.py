import cv2
import pandas as pd
import os
import time

def extractBoudingboxYOLO(csvfile,videofile):
    start_time = time.time()
    csv = str(csvfile)
    video = str(videofile)

    df = pd.read_csv(csv)
    #get unique labels
    labels = df.label.unique()

    #get each df
    dflist = []
    for i in labels:
        dflist.append(df.loc[df.label==i])

    print(len(dflist),'unique labels')

    ## start extract frame
    for i in dflist:
        start_time1 = time.time()
        uniquelabel = i.label.to_list()[0]
        framedf = i.frame.to_list()
        xmin = i.xmin.to_list()
        ymin = i.ymin.to_list()
        xmax = i.xmax.to_list()
        ymax = i.ymax.to_list()

        #make folder
        dir = os.path.dirname(csv)
        newfolder = os.path.join(dir,str('label_'+str(uniquelabel)))
        if not os.path.exists(newfolder):
            os.mkdir(newfolder)

        vid = cv2.VideoCapture(video)
        count = 0
        while vid.isOpened():
            ret, image = vid.read()
            if ret:
                indices = [i for i, x in enumerate(framedf) if x == count] # get all the index
                print(indices)

                for index in indices:
                    x1 = int(xmin[index])
                    x2 = int(xmax[index])
                    y1 = int(ymin[index])
                    y2 = int(ymax[index])
                    img = image[y1:y2,x1:x2]
                    cv2.imwrite(os.path.join(newfolder, str(count)+'_'+str(index)+ '.png'),img)

                count += 1

            else:
                break
        cv2.destroyAllWindows()
        vid.release()
        print("--- %s seconds ---" % (time.time() - start_time1))

    print("--- %s seconds ---" % (time.time() - start_time))

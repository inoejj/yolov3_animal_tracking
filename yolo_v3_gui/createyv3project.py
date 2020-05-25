import os

def createYoloV3_project(projectpath,projectname):

    project_dir = os.path.join(projectpath,projectname)
    imagesPath = os.path.join(project_dir, 'Source_Images')
    modelPath = os.path.join(project_dir,'Model_Weights')
    videoPath = os.path.join(project_dir,'Videos')

    resultPath = os.path.join(imagesPath,'Test_Image_Detection_Results')
    testPath = os.path.join(imagesPath, 'Test_Images')
    trainingPath = os.path.join(imagesPath, 'Training_Images')
    dirlist = [project_dir,imagesPath,modelPath,resultPath,testPath,trainingPath,videoPath]

    for i in dirlist:
        if not os.path.exists(i):
            os.mkdir(i)

    print('Project structure created.')



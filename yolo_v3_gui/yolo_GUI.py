from tkinter import *
from tkinter.filedialog import askopenfilename,askdirectory
from yolo_methods.vottcsv_to_yolotxt import csv_to_yoloready
from yolo_methods.Train_YOLO import runYoloV3
from yolo_methods.Detector import yolov3_detection
from yolo_methods.createyv3project import createYoloV3_project
import os

class FileSelect(Frame):
    def __init__(self,parent=None,fileDescription="",color=None,title=None,lblwidth=None,**kw):
        self.title=title
        self.color = color if color is not None else 'black'
        self.lblwidth = lblwidth if lblwidth is not None else 0
        self.parent=parent
        Frame.__init__(self,master=parent,**kw)
        self.filePath = StringVar()
        self.lblName = Label(self, text=fileDescription,fg=str(self.color),width=str(self.lblwidth),anchor=W)
        self.lblName.grid(row=0,column=0,sticky=W)
        self.entPath = Label(self, textvariable=self.filePath,relief=SUNKEN)
        self.entPath.grid(row=0,column=1)
        self.btnFind = Button(self, text="Browse File",command=self.setFilePath)
        self.btnFind.grid(row=0,column=2)
        self.filePath.set('No file selected')
    def setFilePath(self):
        file_selected = askopenfilename(title=self.title,parent=self.parent)
        if file_selected:
            self.filePath.set(file_selected)
        else:
            self.filePath.set('No file selected')
    @property
    def file_path(self):
        return self.filePath.get()

class Entry_Box(Frame):
    def __init__(self,parent=None,fileDescription="",labelwidth='',status = None,**kw):
        self.status = status if status is not None else NORMAL
        self.labelname = fileDescription
        Frame.__init__(self,master=parent,**kw)
        self.filePath = StringVar()
        self.lblName = Label(self, text=fileDescription,width=labelwidth,anchor=W)
        self.lblName.grid(row=0,column=0)
        self.entPath = Entry(self, textvariable=self.filePath,state=self.status)
        self.entPath.grid(row=0,column=1)

    @property
    def entry_get(self):
        self.entPath.get()
        return self.entPath.get()

    def entry_set(self, val):
        self.filePath.set(val)

    def set_state(self,setstatus):
        self.entPath.config(state=setstatus)

class FolderSelect(Frame):
    def __init__(self,parent=None,folderDescription="",color=None,title=None,lblwidth =None,**kw):
        self.title=title
        self.color = color if color is not None else 'black'
        self.lblwidth = lblwidth if lblwidth is not None else 0
        self.parent = parent
        Frame.__init__(self,master=parent,**kw)
        self.folderPath = StringVar()
        self.lblName = Label(self, text=folderDescription,fg=str(self.color),width=str(self.lblwidth),anchor=W)
        self.lblName.grid(row=0,column=0,sticky=W)
        self.entPath = Label(self, textvariable=self.folderPath,relief=SUNKEN)
        self.entPath.grid(row=0,column=1)
        self.btnFind = Button(self, text="Browse Folder",command=self.setFolderPath)
        self.btnFind.grid(row=0,column=2)
        self.folderPath.set('No folder selected')
    def setFolderPath(self):
        folder_selected = askdirectory(title=str(self.title),parent=self.parent)
        if folder_selected:
            self.folderPath.set(folder_selected)
        else:
            self.folderPath.set('No folder selected')

    @property
    def folder_path(self):
        return self.folderPath.get()

class App:

    def __init__(self):
        scriptdir = os.path.dirname(__file__)
        root = Tk()
        root.title('Simple yoloV3 GUI')
        root.iconbitmap(os.path.join(scriptdir,'yologui.ico'))
        root.minsize(350,400)

        label_createproject = LabelFrame(root,text='1. create yolo project',padx=10,pady=10)
        projectname = Entry_Box(label_createproject,'Project Name','12')
        projectpath = FolderSelect(label_createproject,'Project path',lblwidth='12')
        createproject = Button(label_createproject,text='Create project',command=lambda:createYoloV3_project(projectpath.folder_path,projectname.entry_get))

        label_chgformat = LabelFrame(root,text='2. vott_csv to yolo',padx=10,pady=10)
        vottfolder = FolderSelect(label_chgformat,'Vott folder path',lblwidth='12')
        mainprojectfolder = FolderSelect(label_chgformat,'Project path',lblwidth='12')
        csvfile = FileSelect(label_chgformat,'csv file',lblwidth='12')
        step1button = Button(label_chgformat,text='Change format',command=lambda:csv_to_yoloready(vottfolder.folder_path,mainprojectfolder.folder_path,csvfile.file_path))

        label_train = LabelFrame(root,text='3. Train',padx=10,pady=10)
        projdir = FolderSelect(label_train,'Project path',lblwidth='12')
        weightfile = FileSelect(label_train,'Weight',lblwidth='12')
        step2button = Button(label_train,text='Train',command=lambda:runYoloV3(projdir.folder_path,weightfile.file_path))

        label_detect = LabelFrame(root,text='4. Detect',padx=10,pady=10)
        projdir2 = FolderSelect(label_detect,'Project path',lblwidth='12')
        weightfile2 = FileSelect(label_detect,'Weight',lblwidth='12')
        step3button = Button(label_detect,text='Detect',command=lambda:yolov3_detection(projdir2.folder_path,weightfile2.file_path))

        #organize
        label_createproject.grid(row=0,sticky=W)
        projectname.grid(row=0,sticky=W)
        projectpath.grid(row=1,sticky=W)
        createproject.grid(row=2,sticky=W)

        label_chgformat.grid(row=1,sticky=W)
        vottfolder.grid(row=0,sticky=W)
        mainprojectfolder.grid(row=1,sticky=W)
        csvfile.grid(row=2,sticky=W)
        step1button.grid(row=3,sticky=W)

        label_train.grid(row=2,sticky=W)
        projdir.grid(row=0,sticky=W)
        weightfile.grid(row=1,sticky=W)
        step2button.grid(row=2,sticky=W)

        label_detect.grid(row=3,sticky=W)
        projdir2.grid(row=0,sticky=W)
        weightfile2.grid(row=1,sticky=W)
        step3button.grid(row=2,sticky=W)

        root.mainloop()

if __name__ == '__main__':
    App()


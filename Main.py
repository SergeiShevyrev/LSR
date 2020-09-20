# -*- coding: utf-8 -*-

import sys,os
import imageio as io
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QPushButton,QWidget,\
                    QLineEdit,QFileDialog, QStatusBar,QCheckBox,QMessageBox
from PyQt5.QtCore import QTimer
from PyQt5 import uic

from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from LSR_engine import engineTopo

import pickle

from mygdal_functions0_9 import *

#set proper current directory
current_dir=os.path.abspath(os.path.dirname(sys.argv[0]))
os.chdir(current_dir)


uifile_1 = "LSRMain.ui" # Enter file here.
uifile_2 = "LSR_input.ui" # Enter file here.
uifile_3 = "LSR_configure.ui" # Enter file here.
uifile_4 = "LSR_output.ui" # Enter file here.
uifile_5 = "LSR_about.ui" # Enter file here.

form_1, base_1 = uic.loadUiType(uifile_1)
form_2, base_2 = uic.loadUiType(uifile_2)
form_3, base_3 = uic.loadUiType(uifile_3)
form_4, base_4 = uic.loadUiType(uifile_4)
form_5, base_5 = uic.loadUiType(uifile_5)

class Main(base_1, form_1):
    def __init__(self):
        super(base_1,self).__init__()
        self.setupUi(self)
        #UI windows
        self.InputWindow=[]; 
        self.ProcessWindow=[];
        
        self.SolarAngle=0;
        self.SolarAzimuth=0;
        
        #Program level vars
        self.isSCS_C=True;
        
        #directories
        self.input_dir_name=[];
        self.output_dir_name=[];
        self.srtm_file_name=[];
        self.aoi_file_name=[];
        
        self.products=['NDVI'];
        
        self.bandStacks=['rgb'];
        
        self.was_loaded_config=False;
        self.was_loaded_path=False;
     
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Landsat Shadow Remove v.0.1a");  

        #button procedures connect
        self.BtnSetInputDir.clicked.connect(self.openInputWindow)
        self.BtnSetConfigure.clicked.connect(self.openConfigureWindow)
        self.BtnProcess.clicked.connect(self.openProcessWindow)
        self.BtnAbout.clicked.connect(self.openAboutWindow)
        
        
        
    def dir_in_open_dialogue(self):
        self.input_dir_name = QFileDialog.getExistingDirectory(self,'Please, select a directory, containing Landsat 8 tif files...')    
        self.LblInputDir.setText(self.input_dir_name);
        self.logMessage("Set input dir value to:"+"\n"+self.input_dir_name+"\n");
        #print(self.input_dir_name)
              
    def openInputWindow(self):
        self.InputWindow=Input()
        self.InputWindow.show()
        
    def openConfigureWindow(self):
        self.ConfigureWindow=Configure()
        self.ConfigureWindow.show()
    
    def openProcessWindow(self):
        self.ProcessWindow=Process()
        self.ProcessWindow.show()
    
    def openAboutWindow(self):
        self.AboutWindow=About()
        self.AboutWindow.show()
        
    def setStatusMessage(self,txt): 
        self.statusBar.showMessage(txt);
        
    def logMessage(self, txt):      #outputs message in the status bar
        self.brsLog.setText(self.brsLog.toPlainText() + txt)
       
    def file_open_dialogue(self):
        
        fileName = QFileDialog.getOpenFileName(self,'Please, select SRTM.','','TIF files (*.tif)')
        
        if fileName[0]!='':
            self.filePath.setPlainText(fileName[0])
            self.raster=io.imread(fileName[0])  
            self.build_plot()
        
            
    def showDialog(self,text):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information);
        msgBox.setText(text);
        msgBox.setWindowTitle("Information");
        msgBox.setStandardButtons(QMessageBox.Ok);
        #msgBox.buttonClicked.connect(msgButtonClick)
        returnValue = msgBox.exec()             #without exec message could not be seen
        #if returnValue == QMessageBox.Ok:
        #   print('OK clicked')
    
    def showDialogYesNo(self,text):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information);
        msgBox.setText(text);
        msgBox.setWindowTitle("Please, specify...");
        msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
        #msgBox.buttonClicked.connect(msgButtonClick)
        returnValue = msgBox.exec()             #without exec message could not be seen
        if returnValue == QMessageBox.Ok:
           return True;
        else:
            return False;
            

class Input(base_2, form_2):
    def __init__(self):
        super(base_2,self).__init__()
        self.setupUi(self)
        
        #button procedures connect
        self.BtnSetInputDir.clicked.connect(self.dir_in_open_dialogue)
        self.BtnSetSRTMfile.clicked.connect(self.srtm_open_dialogue)
        self.BtnSetAOIfile.clicked.connect(self.aoi_open_dialogue)
        self.BtnSetOutputDir.clicked.connect(self.dir_out_open_dialogue)
        self.BtnOk.clicked.connect(self.closeWin)
        
        #load data from the configuration file if exists  
        if main.was_loaded_path==False:
            if os.path.isfile('configuration.file'):
                if(main.showDialogYesNo('Load previous configuration?')):
                    main.was_loaded_path=True;
                    with open('configuration.file', 'rb') as f:
                        [main.input_dir_name, main.output_dir_name, main.srtm_file_name, main.aoi_file_name] = pickle.load(f)
            
        #set text fields according to main class vars
        if(len(main.input_dir_name)>0):
            self.txtInputDir.setText(main.input_dir_name);
        if(len(main.output_dir_name)>0):
            self.txtOutputDir.setText(main.output_dir_name);
        if(len(main.srtm_file_name)>0):
            self.txtSRTMfile.setText(main.srtm_file_name);
        if(len(main.aoi_file_name)>0):
            self.txtAOIfile.setText(main.aoi_file_name);
        
        #report
        main.logMessage("Input dialogue was opened..."+"\n");
        
    def dir_in_open_dialogue(self):
        input_dir_name = QFileDialog.getExistingDirectory(self,'Please, select a directory, containing Landsat 8 tif files...')    
        if input_dir_name!='':
            #main.LblInputDir.setText(input_dir_name);
            #main.logMessage("Set input dir value to:"+"\n"+main.input_dir_name+"\n");
            self.txtInputDir.setText(input_dir_name);
    
    def dir_out_open_dialogue(self):
        output_dir_name = QFileDialog.getExistingDirectory(self,'Please, select a directory for the file output...')    
        if output_dir_name!='':
            #main.LblOutputDir.setText(main.output_dir_name);
            #main.logMessage("Set output dir value to:"+"\n"+main.output_dir_name+"\n");
            self.txtOutputDir.setText(output_dir_name);
    
    def srtm_open_dialogue(self):
        fileSRTMName = QFileDialog.getOpenFileName(self,'Please, select SRTM file','','TIF files (*.tif)')
          
        if fileSRTMName[0]!='':
            self.txtSRTMfile.setText(fileSRTMName[0])
            #main.srtm_file_name=fileSRTMName[0];  
            #main.logMessage("SRTM file was set as:"+fileSRTMName[0]+"\n");
    
    def aoi_open_dialogue(self):
        fileAOIName = QFileDialog.getOpenFileName(self,'Please, select AOI file','','SHP files (*.shp)')
          
        if fileAOIName[0]!='':
            self.txtAOIfile.setText(fileAOIName[0])
            #main.aoi_file_name=fileAOIName[0];
            #main.logMessage("SRTM file was set as:"+fileAOIName[0]+"\n");
    
    def closeWin(self):
        main.logMessage("Input dialogue was closed..."+"\n");
        #get directory values from the text fields
        main.input_dir_name=self.txtInputDir.text();
        main.LblInputDir.setText(main.input_dir_name);
        main.logMessage("Set input dir value to:"+"\n"+main.input_dir_name+"\n");
        
        main.output_dir_name=self.txtOutputDir.text();
        main.LblOutputDir.setText(main.output_dir_name);
        main.logMessage("Set output dir value to:"+"\n"+main.output_dir_name+"\n");
        
        main.srtm_file_name=self.txtSRTMfile.text();
        main.logMessage("SRTM file was set as:"+main.srtm_file_name+"\n");
        
        main.aoi_file_name=self.txtAOIfile.text();
        main.logMessage("SRTM file was set as:"+main.aoi_file_name+"\n");
        
        #save products data to pickle file
        with open('configuration.file', 'wb') as f:
            pickle.dump([main.input_dir_name, main.output_dir_name, main.srtm_file_name, main.aoi_file_name], f)
        
        self.close();

class Configure(base_3, form_3):
    def __init__(self):
        super(base_3,self).__init__()
        self.setupUi(self)
         #load data from the configuration file if exists    
        if main.was_loaded_config==False:
            if os.path.isfile('products.file'):
                if(main.showDialogYesNo('Load previous configuration?')):
                    main.was_loaded_config=True;
                    with open('products.file', 'rb') as f:
                        [main.bandStacks, main.products, main.SolarAzimuth, main.SolarAngle] = pickle.load(f)
                        
        #button procedures connect
        self.BtnMarkAll.clicked.connect(self.markAll)
        self.BtnUnMarkAll.clicked.connect(self.unMarkAll)
        self.BtnOk.clicked.connect(self.closeWin)
        self.chkSCS_C.stateChanged.connect(lambda:self.btnstate(self.chkSCS_C))
        self.chkNone.stateChanged.connect(lambda:self.btnstate(self.chkNone))
        
        #report
        main.logMessage("Configuration dialogue was opened..."+"\n");
        #mark previously selected chekboxes
        if (main.isSCS_C==True):
            self.chkSCS_C.setChecked(True);
        else:
            self.chkNone.setChecked(True);
        for chbox in self.findChildren(QCheckBox):
            if chbox.isChecked()==False:
                if (chbox.text() in main.products) or \
                (chbox.text() in main.bandStacks):
                    chbox.setChecked(True);
        
        self.txtSolarAzimuth.setText(str(main.SolarAzimuth));
        self.txtSolarAngle.setText(str(main.SolarAngle));
    
    def btnstate(self,b):
        if b.text() == "None":
            if b.isChecked() == True:
                self.chkSCS_C.setChecked(False);
                main.isSCS_C=False;
        if b.text() == "SCS+C":
            if b.isChecked() == True:
                self.chkNone.setChecked(False);
                main.isSCS_C=True;
    
    def markAll(self,b):
        """
        ww=self.findChildren();
        for w in ww:
            print(w.getName());
        """
        for chbox in self.findChildren(QCheckBox):
            
            if chbox.isChecked()==False:
              if chbox.text() != "None" and chbox.text() != "SCS+C":      
                  chbox.setChecked(True); 
            
            if chbox.text() == "SCS+C":
                chbox.setChecked(True);
            if chbox.text() == "None":
                chbox.setChecked(False);   #opposite values due to btnstate function
    
    def unMarkAll(self,b):
        for chbox in self.findChildren(QCheckBox):
            if chbox.isChecked()==True:
                chbox.setChecked(False); 
                      
           
    def closeWin(self):
        main.logMessage("Configuration dialogue was closed"+"\n");
        #check chekboxes state and create product string here!!!
        """
        looking for the possible regimes
        main.products=['NDVI', 'AOI', 'HA', 'CM', 'CA', 'PC'];
        main.bandStacks=['rgb', '742', '652', '453', '642', '764','765'];
        """
        main.isSCS_C=self.chkSCS_C.isChecked();
        main.products=[]; main.bandStacks=[];
        for chbox in self.findChildren(QCheckBox):
            if chbox.isChecked()==True:
                if chbox.text() == "SCS+C":
                    main.isSCS_C = True;
                if chbox.text() == "NDVI":
                    main.products.append("NDVI");
                if chbox.text() == "AOI":
                    main.products.append("AOI");
                if chbox.text() == "HA":
                    main.products.append("HA");
                if chbox.text() == "CM":
                    main.products.append("CM");
                if chbox.text() == "CA":
                    main.products.append("CA");
                if chbox.text() == "PC":
                    main.products.append("PC");
                if chbox.text() == "truecolor RGB":
                    main.bandStacks.append("rgb");
                if chbox.text() == "742":
                    main.bandStacks.append("742");
                if chbox.text() == "652":
                    main.bandStacks.append("652");
                if chbox.text() == "453":
                    main.bandStacks.append("453");
                if chbox.text() == "642":
                    main.bandStacks.append("642");
                if chbox.text() == "764":
                    main.bandStacks.append("764");
                if chbox.text() == "765":
                    main.bandStacks.append("765");
        try:
            main.SolarAzimuth=float(self.txtSolarAzimuth.text());
        except:
            main.showDialog('Only number could be solar azimuth values')
        try:
            main.SolarAngle=float(self.txtSolarAngle.text());
        except:
            main.showDialog('Only numbers could be solar elevation angle values');
        print(main.bandStacks);
        print(main.products);
        
        #save products data to pickle file
        with open('products.file', 'wb') as f:
            pickle.dump([main.bandStacks, main.products, main.SolarAzimuth, main.SolarAngle], f)
        
        self.close();               

class Process(base_4, form_4):
    def __init__(self):
        super(base_4,self).__init__()
        self.setupUi(self)
        
        #connect buttons
        self.BtnSTART.clicked.connect(self.startProcess);
        self.BtnClose.clicked.connect(self.closeWin);
        #pb
        self.progressBar.setValue(0);
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.timerDo)
        
        
        self.completed=0;
    
    def timerDo(self):
        while self.completed < 100:
            self.completed += 0.01
            self.progressBar.setValue(self.completed)
        
    def startProcess(self):
        #
        self.completed=10;
        self.progressBar.setValue(10);
        self.timer.start(1000);
        if self.checkBeforeConvert()==False:
            main.showDialog("Input/output directories haven\'t set!\n Task was cancelled \n");
            main.logMessage("Input/output directories haven\'t set! Task was cancelled \n");
        else:
            self.BtnClose.setEnabled(False);
            print('is_topocorrection={}'.format(main.isSCS_C));
            result=engineTopo(in_path=main.input_dir_name,out_path=main.output_dir_name,\
                       shpfilepath=main.aoi_file_name,drm_filepath=main.srtm_file_name,\
               products=main.products,bandStacks=main.bandStacks,is_topocorrection=main.isSCS_C,\
               SunElevation=main.SolarAngle, SunAzimuth=main.SolarAzimuth,fileext="tif")
            if result==-1:
                main.showDialog("Process ended with errors");
            else:
                self.completed=100;
                self.progressBar.setValue(100);
                main.showDialog("Process succesfully done");
            '''self.isSCS_C
                    self.input_dir_name=[];
        self.output_dir_name=[];
        self.srtm_file_name=[];
        self.aoi_file_name=[];
        
        self.products=['NDVI'];
        
        self.bandStacks=['rgb'];
            '''
            
            self.timer.stop();
            self.BtnClose.setEnabled(True);

    def checkBeforeConvert(self):
        #check if output directory exist
        #check if any of output products are selected
        if(os.path.isdir(str(main.input_dir_name))==False):
            main.logMessage("Warning! No input directory..."+"\n");
            return False;
        if(os.path.isdir(str(main.output_dir_name))==False):
            main.logMessage("Warning! No output directory..."+"\n");
            return False;
        if(os.path.isfile(main.srtm_file_name)):
            main.logMessage("Warning! No SRTM file selected. Topocorrection won\'t be available..."+"\n");
        if(os.path.isfile(main.aoi_file_name)):
            main.logMessage("Warning! No SHP file selected. Crop won\'t be available..."+"\n");
        if(len(main.products)==0):
            main.logMessage("Warning! No products were selected..."+"\n");
        if(len(main.bandStacks)==0):
            main.logMessage("Warning! No band stacks were selected..."+"\n");
        return True;

    def closeWin(self):
        main.logMessage("Process window was closed..."+"\n");
        self.close(); 
        
              
   
class About(base_5, form_5):
    def __init__(self):
        super(base_5,self).__init__()
        self.setupUi(self)
        
        #connect buttons
        self.BtnClose.clicked.connect(self.closeWin)
    
    def closeWin(self):
        main.logMessage("About window was closed..."+"\n");
        self.close();
                                

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
    
    
    
    """    
    def clearInput(self):
        self.XY=[[],[]]
        self.build_plot()
     
    def onClick(self,event):
        #print(event.xdata,event.ydata)
        print("Xclick="+str(event.xdata))
        print("Yclick="+str(event.ydata))
        self.XY[0].append(int(event.xdata))
        self.XY[1].append(int(event.ydata))
        self.drawLine()
        
    def drawLine(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(self.raster,cmap=plt.get_cmap('gray'))
        ax.plot(self.XY[0],self.XY[1],'ro--')
        #print(self.XY[0],self.XY[1])
        self.canvas.draw()
    """ 
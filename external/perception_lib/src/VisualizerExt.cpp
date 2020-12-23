#include "VisualizerExt.h"

boost::shared_ptr<pcl::visualization::PCLVisualizer> VisualizerExt::viewer;
VisualizerExtBuffer VisualizerExt::buffer;
bool VisualizerExt::enabled;
int VisualizerExt::count;
boost::mutex VisualizerExt::bufferMutex;
bool VisualizerExt::stopFlag;
bool VisualizerExt::mouseCallbackMode;
bool VisualizerExt::keyboardCallbackMode;
boost::function<void(int, int, int)> VisualizerExt::mouseCallbackExt;
boost::function<void(int)> VisualizerExt::keyboardCallbackExt;

VisualizerExtBuffer::VisualizerExtBuffer(){
    bufferXYZ.push_back(std::vector<PointCloudExt<pcl::PointXYZ> >()); bufferXYZ.push_back(std::vector<PointCloudExt<pcl::PointXYZ> >());
    bufferXYZRGB.push_back(std::vector<PointCloudExt<pcl::PointXYZRGB> >()); bufferXYZRGB.push_back(std::vector<PointCloudExt<pcl::PointXYZRGB> >());
    bufferXYZRGBNormal.push_back(std::vector<PointCloudExt<pcl::PointXYZRGBNormal> >()); bufferXYZRGBNormal.push_back(std::vector<PointCloudExt<pcl::PointXYZRGBNormal> >());
    bufferNormals.push_back(std::vector<PointCloudExt<pcl::PointXYZRGBNormal> >()); bufferNormals.push_back(std::vector<PointCloudExt<pcl::PointXYZRGBNormal> >());
    bufferOBB.push_back(std::vector<PointCloudExtParams::OBB>()); bufferOBB.push_back(std::vector<PointCloudExtParams::OBB>());
    bufferString.push_back(std::vector<std::string>()); bufferString.push_back(std::vector<std::string>());
    bufferStringPos.push_back(std::vector<pcl::PointXYZ>()); bufferStringPos.push_back(std::vector<pcl::PointXYZ>());
    bufferMat.push_back(std::vector<MatExt>()); bufferMat.push_back(std::vector<MatExt>());
    bufferArrow.push_back(std::vector<VisualizerExtBuffer::ArrowData>()); bufferArrow.push_back(std::vector<VisualizerExtBuffer::ArrowData>());
    bufferMesh.push_back(std::vector<pcl::PolygonMesh>()); bufferMesh.push_back(std::vector<pcl::PolygonMesh>());
    bufferTextureMesh.push_back(std::vector<pcl::TextureMesh>()); bufferTextureMesh.push_back(std::vector<pcl::TextureMesh>());
    bufferLine.push_back(std::vector<VisualizerExtBuffer::LineData>()); bufferLine.push_back(std::vector<VisualizerExtBuffer::LineData>());
}

void VisualizerExtBuffer::swapBuffer(){
    bufferMutex.lock();
    bufferXYZ.erase(bufferXYZ.begin());
    bufferXYZRGB.erase(bufferXYZRGB.begin());
    bufferXYZRGBNormal.erase(bufferXYZRGBNormal.begin());
    bufferNormals.erase(bufferNormals.begin());
    bufferOBB.erase(bufferOBB.begin());
    bufferString.erase(bufferString.begin());
    bufferStringPos.erase(bufferStringPos.begin());
    bufferMat.erase(bufferMat.begin());
    bufferArrow.erase(bufferArrow.begin());
    bufferMesh.erase(bufferMesh.begin());
    bufferTextureMesh.erase(bufferTextureMesh.begin());
    bufferLine.erase(bufferLine.begin());
    bufferXYZ.push_back(std::vector<PointCloudExt<pcl::PointXYZ> >());
    bufferXYZRGB.push_back(std::vector<PointCloudExt<pcl::PointXYZRGB> >());
    bufferXYZRGBNormal.push_back(std::vector<PointCloudExt<pcl::PointXYZRGBNormal> >());
    bufferNormals.push_back(std::vector<PointCloudExt<pcl::PointXYZRGBNormal> >());
    bufferOBB.push_back(std::vector<PointCloudExtParams::OBB>());
    bufferString.push_back(std::vector<std::string>());
    bufferStringPos.push_back(std::vector<pcl::PointXYZ>());
    bufferMat.push_back(std::vector<MatExt>());
    bufferArrow.push_back(std::vector<VisualizerExtBuffer::ArrowData>());
    bufferMesh.push_back(std::vector<pcl::PolygonMesh>());
    bufferTextureMesh.push_back(std::vector<pcl::TextureMesh>());
    bufferLine.push_back(std::vector<VisualizerExtBuffer::LineData>());
    bufferMutex.unlock();
}

void VisualizerExt::mouseCallback(int event, int x, int y, int flags, void *param){
    if(mouseCallbackMode){
        mouseCallbackExt(x,y, event);
    }
}

void VisualizerExt::keyboardCallback(int kb){
    if(keyboardCallbackMode){
        keyboardCallbackExt(kb);
    }
}

void VisualizerExt::setMouseCallback(boost::function<void (int, int, int)> &callback){
    mouseCallbackMode = true;
    mouseCallbackExt = callback;
}

void VisualizerExt::setKeyboardCallback(boost::function<void (int)> &callback){
    keyboardCallbackMode = true;
    keyboardCallbackExt = callback;
}

void VisualizerExtBuffer::updateViewer(boost::shared_ptr<pcl::visualization::PCLVisualizer>& viewer){
    bufferMutex.lock();
    viewer->removeAllPointClouds();
    viewer->removeAllShapes();
    for(int i=0; i<bufferXYZ[0].size(); i++){
        viewer->addPointCloud(bufferXYZ[0][i].getPtrCopy(), "XYZ" + boost::to_string(i));
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, bufferXYZ[0][i].visualizerSize, "XYZ" + boost::to_string(i));
    }
    for(int i=0; i<bufferXYZRGB[0].size(); i++){
        viewer->addPointCloud(bufferXYZRGB[0][i].getPtrCopy(), "XYZRGB" + boost::to_string(i));
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, bufferXYZRGB[0][i].visualizerSize, "XYZRGB" + boost::to_string(i));
    }
    for(int i=0; i<bufferXYZRGBNormal[0].size(); i++){
        PointCloudExt<pcl::PointXYZRGB> convert;
        pcl::copyPointCloud(bufferXYZRGBNormal[0][i], convert);
        convert.visualizerSize = bufferXYZRGBNormal[0][i].visualizerSize;
        viewer->addPointCloud(convert.getPtrCopy(), "XYZRGBNormal" + boost::to_string(i));
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, convert.visualizerSize, "XYZRGBNormal" + boost::to_string(i));
    }
    for(int i=0; i<bufferNormals[0].size(); i++){
        if(!bufferNormals[0][i].size()) continue;
        viewer->addPointCloudNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> (bufferNormals[0][i].getPtrCopy(), bufferNormals[0][i].getPtrCopy(), 10, 0.05, "Normal" + boost::to_string(i));
    }
    for(int i=0; i<bufferOBB[0].size(); i++){
        viewer->addCube(bufferOBB[0][i].bboxTransform, bufferOBB[0][i].bboxQuaternion, bufferOBB[0][i].xDist, bufferOBB[0][i].yDist, bufferOBB[0][i].zDist, "OBB" + boost::to_string(i));
    }
    for(int i=0; i<bufferString[0].size(); i++){
        viewer->addText3D(bufferString[0][i], bufferStringPos[0][i], 0.02, 255,255,255, "String" + boost::to_string(i));
    }
    for(int i=0; i<bufferMat[0].size(); i++){
        if(bufferMat[0][i].name.length() > 0){
            cv::imshow(bufferMat[0][i].name, bufferMat[0][i]);
        }else{
            cv::imshow("Image" + boost::to_string(i), bufferMat[0][i]);
            cv::setMouseCallback("Image" + boost::to_string(i), VisualizerExt::mouseCallback, 0);
        }
    }
    for(int i=0; i<bufferArrow[0].size(); i++){
        if(bufferArrow[0][i].head)
        viewer->addArrow(bufferArrow[0][i].p2, bufferArrow[0][i].p1, (float)bufferArrow[0][i].r/255, (float)bufferArrow[0][i].g/255, (float)bufferArrow[0][i].b/255, false, "Arrow" + boost::to_string(i));
        else
        viewer->addLine(bufferArrow[0][i].p2, bufferArrow[0][i].p1, (float)bufferArrow[0][i].r/255, (float)bufferArrow[0][i].g/255, (float)bufferArrow[0][i].b/255, "Arrow" + boost::to_string(i));
    }
    for(int i=0; i<bufferMesh[0].size(); i++){
        viewer->addPolygonMesh(bufferMesh[0][i], "Mesh" + boost::to_string(i));
    }
    for(int i=0; i<bufferTextureMesh[0].size(); i++){
        viewer->addTextureMesh(bufferTextureMesh[0][i], "TextureMesh" + boost::to_string(i));
    }
    if(bufferLine[0].size()){
        vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New ();
        vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New ();
        vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New ();
        colors->SetNumberOfComponents (3);
        colors->SetName ("Colors");
        vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New ();
        for(int i=0; i<bufferLine[0].size(); i++){
            VisualizerExtBuffer::LineData& line = bufferLine[0][i];
            unsigned char color [3] = {(unsigned char)line.r,(unsigned char)line.g,(unsigned char)line.b};
            colors->InsertNextTupleValue (color);
            colors->InsertNextTupleValue (color);
            points->InsertNextPoint (line.p1.data);
            points->InsertNextPoint (line.p2.data);
            vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New ();
            polyLine->GetPointIds ()->SetNumberOfIds (2);
            polyLine->GetPointIds ()->SetId (0, points->GetNumberOfPoints ()-2);
            polyLine->GetPointIds ()->SetId (1, points->GetNumberOfPoints ()-1);
            cells->InsertNextCell (polyLine);
        }
        polyData->SetPoints (points);
        polyData->SetLines (cells);
        polyData->GetPointData ()->SetScalars (colors);
        viewer->addModelFromPolyData (polyData,"Line" + boost::to_string(1));
    }
    bufferMutex.unlock();
}

void VisualizerExt::loop(){
    if (!viewer->wasStopped ()) {
        buffer.updateViewer(viewer);
        viewer->spinOnce(10);
        count = 0;
    }
}

void VisualizerExt::stop(){
    viewer->close();
}

void VisualizerExt::saveScreenshot(std::string file){
    viewer->saveScreenshot(file);
}

cv::Mat VisualizerExt::getRenderedImage(){
    vtkSmartPointer<vtkRenderWindow> render = viewer->getRenderWindow();
    unsigned char* pixels = render->GetRGBACharPixelData(0, 0, render->GetSize()[0] - 1, render->GetSize()[1] - 1, 1);
    cv::Mat image = cv::Mat(render->GetSize()[1], render->GetSize()[0], CV_8UC4, pixels).clone();
    //cv::cvtColor(image, image, cv::COLOR_RGBA2BGRA);
    cv::cvtColor(image, image, cv::COLOR_RGBA2BGR);
    cv::flip(image, image, 0);
    return image;
}

void VisualizerExt::start(){

    // Set signal handler
    //signal(SIGINT, &VisualizerExt::signalHandler);

    // Start the visualizer
    viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer> (new pcl::visualization::PCLVisualizer ("PCL Visualizer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addCoordinateSystem (1.0);
    enabled = true;

    // Loop
    //ros::AsyncSpinner spinner(8);
    //spinner.start();
    // while (!viewer->wasStopped ()) {
    //     int kb = cv::waitKey(15);
    //     if(kb != 255) VisualizerExt::keyboardCallback(kb);
    //     buffer.updateViewer(viewer);
    //     viewer->spinOnce(10);
    //     count = 0;
    //     if(stopFlag) break;
    // }

    // // Stop
    // viewer->close();
    //spinner.stop();
}

void VisualizerExt::swapBuffer(){
    if(!enabled) return;
    buffer.swapBuffer();
}

void VisualizerExt::addArrow(VisualizerExtBuffer::ArrowData arrow){
    if(!enabled) return;
    bufferMutex.lock();
    buffer.bufferArrow[1].push_back(arrow);
    bufferMutex.unlock();
    count++;
}

void VisualizerExt::addLine(VisualizerExtBuffer::LineData line){
    if(!enabled) return;
    bufferMutex.lock();
    buffer.bufferLine[1].push_back(line);
    bufferMutex.unlock();
    count++;
}

void VisualizerExt::addPointCloud(PointCloudExt<pcl::PointXYZ> cloud, int size){
    if(!enabled) return;
    bufferMutex.lock();
    cloud.visualizerSize = size;
    buffer.bufferXYZ[1].push_back(cloud);
    bufferMutex.unlock();
    count++;
}

void VisualizerExt::addPointCloud(PointCloudExt<pcl::PointXYZRGB> cloud, int size){
    if(!enabled) return;
    bufferMutex.lock();
    cloud.visualizerSize = size;
    buffer.bufferXYZRGB[1].push_back(cloud);
    bufferMutex.unlock();
    count++;
}

void VisualizerExt::addPointCloud(PointCloudExt<pcl::PointXYZRGBNormal> cloud, int size){
    if(!enabled) return;
    bufferMutex.lock();
    cloud.visualizerSize = size;
    buffer.bufferXYZRGBNormal[1].push_back(cloud);
    bufferMutex.unlock();
    count++;
}

void VisualizerExt::addPoint(pcl::PointXYZ point, int size){
    if(!enabled) return;
    bufferMutex.lock();
    PointCloudExt<pcl::PointXYZ> cloud;
    cloud.push_back(point);
    cloud.visualizerSize = size;
    buffer.bufferXYZ[1].push_back(cloud);
    bufferMutex.unlock();
    count++;
}

void VisualizerExt::addPoint(pcl::PointXYZRGB point, int size){
    if(!enabled) return;
    bufferMutex.lock();
    PointCloudExt<pcl::PointXYZRGB> cloud;
    cloud.push_back(point);
    cloud.visualizerSize = size;
    buffer.bufferXYZRGB[1].push_back(cloud);
    bufferMutex.unlock();
    count++;
}

void VisualizerExt::addPoint(pcl::PointXYZRGBNormal point, int size){
    if(!enabled) return;
    bufferMutex.lock();
    PointCloudExt<pcl::PointXYZRGBNormal> cloud;
    cloud.push_back(point);
    cloud.visualizerSize = size;
    buffer.bufferXYZRGBNormal[1].push_back(cloud);
    bufferMutex.unlock();
    count++;
}

void VisualizerExt::addPoint(float x, float y, float z, int r, int g, int b, int size){
    if(!enabled) return;
    bufferMutex.lock();
    pcl::PointXYZRGB point(r, g, b);
    point.x = x; point.y = y; point.z = z;
    PointCloudExt<pcl::PointXYZRGB> cloud;
    cloud.push_back(point);
    cloud.visualizerSize = size;
    buffer.bufferXYZRGB[1].push_back(cloud);
    bufferMutex.unlock();
    count++;
}

void VisualizerExt::addNormals(PointCloudExt<pcl::PointXYZRGBNormal> cloud){
    if(!enabled) return;
    bufferMutex.lock();
    buffer.bufferNormals[1].push_back(cloud);
    bufferMutex.unlock();
    count++;
}

void VisualizerExt::addOBB(PointCloudExtParams::OBB obb){
    if(!enabled) return;
    bufferMutex.lock();
    buffer.bufferOBB[1].push_back(obb);
    bufferMutex.unlock();
    count++;
}

void VisualizerExt::addText(std::string text, pcl::PointXYZ position){
    if(!enabled) return;
    bufferMutex.lock();
    count++;
    buffer.bufferString[1].push_back(text);
    buffer.bufferStringPos[1].push_back(position);
    bufferMutex.unlock();
}

void VisualizerExt::addImage(MatExt img, float resize, std::string name){
    if(!enabled) return;
    bufferMutex.lock();
    count++;
    if(resize) cv::resize(img, img, cv::Size(0,0), resize, resize);
    img.name = name;
    buffer.bufferMat[1].push_back(img);
    bufferMutex.unlock();
}

void VisualizerExt::addImage(MatExt img, float resize){
    if(!enabled) return;
    bufferMutex.lock();
    count++;
    if(resize) cv::resize(img, img, cv::Size(0,0), resize, resize);
    buffer.bufferMat[1].push_back(img);
    bufferMutex.unlock();
}

void VisualizerExt::addMesh(pcl::PolygonMesh mesh){
    if(!enabled) return;
    bufferMutex.lock();
    buffer.bufferMesh[1].push_back(mesh);
    bufferMutex.unlock();
    count++;
}

void VisualizerExt::addTextureMesh(pcl::TextureMesh mesh){
    if(!enabled) return;
    bufferMutex.lock();
    buffer.bufferTextureMesh[1].push_back(mesh);
    bufferMutex.unlock();
    count++;
}

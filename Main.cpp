#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <locale>
#include <fstream>
#include <sstream>
#include <limits>
#include <map>
#include <algorithm>
#include <string>
#include <vector>

#include <GL/glew.h>

#ifdef __APPLE__
    #include <OpenGL/glu.h>
#else
    #include <GL/glut.h>
#endif

#include <GLFW/glfw3.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifdef _WIN32
#ifdef __cplusplus
extern "C" {
#endif
#include <windows.h>

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#include <mmsystem.h>
#ifdef __cplusplus
}
#endif
#pragma comment(lib, "winmm.lib")
#else
#if defined(__unix__) || defined(__APPLE__)
#include <sys/time.h>
#else
#include <ctime>
#endif
#endif

#ifdef _WIN32
    #pragma comment(linker, "/SUBSYSTEM:windows /ENTRY:mainCRTStartup")
#endif

#ifdef _MSC_VER
    #pragma warning(disable:4996)
#endif

std::string version = "0.9.3a";
//display properties
int width = 768;
int height = 768;
float far_ = 100.0;
float near_ = 1.0;
float camRatio = 0.3;

//face tracking properties
int minFaceSize = 80; // in pixel. The smaller it is, the further away you can go
float f = 500; //804.71_
float eyesGap = 7; //cm
float pixelNbrPerCm = 10.0;
int videoCaptureDevice = 0;
int lockedHeadDistance = 30.0;

//flags
bool drawWireframe = false;
bool invertCam = false;
bool displayCam = true;
bool displayDetection = true;
bool projectionMode = true;
bool lockZ = false;
bool textureFiltering = false;

//paths
std::string configPath = "data/config.cfg";
std::string modelPath = "data/models/su.obj";
cv::String face_cascade_name = "data/haarcascade_frontalface_alt.xml";

//globals
float filteredVal = 0;
float k;
float cx;
float cy;
int camWidth;
int camHeight;
GLdouble glCamX;
GLdouble glCamY;
GLdouble glCamZ = -1;
typedef struct {
    GLuint vb_id;  // vertex buffer id
    int numTriangles;
    size_t material_id;
} DrawObject;
std::vector<DrawObject> gDrawObjects;
cv::CascadeClassifier face_cascade;
cv::VideoCapture* capture = NULL;
cv::Mat frame;
GLFWwindow* window;

//config functions
bool readConfig(std::string cfgPath) {
    using namespace std;
    string currentLine;
    string currentProperty;
    ifstream cfgFile;
    size_t pos;
    map <string, string> cfg;
    cfgFile.open(cfgPath);
    if (!cfgFile.is_open()) {
        cout << "Can't open config file. Continuing with default settings.\n";
        return false;
    }
    cout << "Loading config:\n";
    while (getline(cfgFile, currentLine)) {
        //cout << currentLine << endl;
        currentLine.erase(remove_if(
            begin(currentLine), end(currentLine),
            [l = locale{}](auto ch) { return isspace(ch, l); }
        ), end(currentLine));
        pos = currentLine.find("=");
        if (pos == string::npos) {
            cfgFile.close();
            return false;
        }
        currentProperty = currentLine.substr(0, pos);
        currentLine.erase(0, pos + 1);
        cfg[currentProperty] = currentLine;
        cfg.insert( pair<string, string>(currentProperty, currentLine) );
    }
    cfgFile.close();

    for (auto it = cfg.begin(); it != cfg.end(); ++it)///вывод на экран
    {
        cout << it->first << " = " << it->second << endl;
    }
    
    //setting modelPath value
    map <string, string>::iterator iter = cfg.find("modelPath");
    if (iter == cfg.end())
        return false;
    modelPath = iter->second;
    //setting videoCaptureDevice value
    iter = cfg.find("videoCaptureDevice");
    if (iter == cfg.end())
        return false;
    try {
        videoCaptureDevice = stoi(iter->second);
    }
    catch (...) {
        return false;
    }
    //setting face_cascade_name value
    iter = cfg.find("cascadePath");
    if (iter == cfg.end())
        return false;
    face_cascade_name = iter->second;
    //setting far_ value
    iter = cfg.find("cameraFar");
    if (iter == cfg.end())
        return false;
    try {
        far_ = stof(iter->second);
    }
    catch (...) {
        return false;
    }
    //setting near_ value
    iter = cfg.find("cameraNear");
    if (iter == cfg.end())
        return false;
    try {
        near_ = stof(iter->second);
    }
    catch (...) {
        return false;
    }
    //setting camRatio value
    iter = cfg.find("cameraRatio");
    if (iter == cfg.end())
        return false;
    try {
        camRatio = stof(iter->second);
    }
    catch (...) {
        return false;
    }
    //setting minFaceSize value
    iter = cfg.find("minFaceSize");
    if (iter == cfg.end())
        return false;
    try {
        minFaceSize = stoi(iter->second);
    }
    catch (...) {
        return false;
    }
    //setting f value
    iter = cfg.find("f");
    if (iter == cfg.end())
        return false;
    try {
        f = stof(iter->second);
    }
    catch (...) {
        return false;
    }
    //setting eyesGap value
    iter = cfg.find("eyesGapInCm");
    if (iter == cfg.end())
        return false;
    try {
        eyesGap = stof(iter->second);
    }
    catch (...) {
        return false;
    }
    //setting pixelNbrPerCm value
    iter = cfg.find("pixelPerInch");
    if (iter != cfg.end()) {
        try {
            pixelNbrPerCm = stof(iter->second)*2.54f;
        }
        catch (...) { ; }
    }
    iter = cfg.find("pixelPerCm");
    if (iter == cfg.end())
        return false;
    try {
        pixelNbrPerCm = stof(iter->second);
    }
    catch (...) {
        return false;
    }
    //setting lockedHeadDistance value
    iter = cfg.find("lockedHeadDistance");
    if (iter != cfg.end()) {
        try {
            lockedHeadDistance = stoi(iter->second);
        }
        catch (...) {
            ;
        }
    }
    //setting drawWireframe value
    iter = cfg.find("drawWireframe");
    string lowercase = iter->second;
    if (iter != cfg.end()) {
        transform(lowercase.begin(), lowercase.end(), lowercase.begin(),
            [](unsigned char c) { return tolower(c); });
        if (lowercase == "true" || lowercase == "1")
            drawWireframe = true;
        else if (lowercase == "false" || lowercase == "0")
            drawWireframe = false;
    }
    //setting projectionMode value
    iter = cfg.find("projectionMode");
    if (iter != cfg.end()) {
        lowercase = iter->second;
        transform(lowercase.begin(), lowercase.end(), lowercase.begin(),
            [](unsigned char c) { return tolower(c); });
        if (lowercase == "true" || lowercase == "1")
            projectionMode = true;
        else if (lowercase == "false" || lowercase == "0")
            projectionMode = false;
    }
    //setting lockZ value
    iter = cfg.find("lockZ");
    if (iter != cfg.end()) {
        lowercase = iter->second;
        transform(lowercase.begin(), lowercase.end(), lowercase.begin(),
            [](unsigned char c) { return tolower(c); });
        if (lowercase == "true" || lowercase == "1")
            lockZ = true;
        else if (lowercase == "false" || lowercase == "0")
            lockZ = false;
    }
    //setting textureFiltering value
    iter = cfg.find("textureFiltering");
    if (iter != cfg.end()) {
        lowercase = iter->second;
        transform(lowercase.begin(), lowercase.end(), lowercase.begin(),
            [](unsigned char c) { return tolower(c); });
        if (lowercase == "true" || lowercase == "1")
            textureFiltering = true;
        else if (lowercase == "false" || lowercase == "0")
            textureFiltering = false;
    }
    return true;
}

//face tracking functions
int countCameras()
{
    cv::VideoCapture temp_camera;
    int maxTested = 10;
    for (int i = 0; i < maxTested; i++) {
        cv::VideoCapture temp_camera(i);
        bool res = (!temp_camera.isOpened());
        temp_camera.release();
        if (res)
        {
            return i;
        }
    }
    return maxTested;
}
float pixelToCm(int size)
{
    return (float)size / pixelNbrPerCm;
}
void setGlCamera()
{
    if (projectionMode)
    {
        /* SKEWED FRUSTRUM / OFF-AXIS PROJECTION
        ** My implementation is based on the following paper:
        ** Name:   Generalized Perspective Projection
        ** Author: Robert Kooima
        ** Date:   August 2008, revised June 2009
        */

        //-- space corners coordinates
        float pa[3] = { -cx,-cy,0 };
        float pb[3] = { cx,-cy,0 };
        float pc[3] = { -cx,cy,0 };
        float pe[3] = { glCamX,glCamY,glCamZ };
        //-- space points
        cv::Vec3f Pa(pa);
        cv::Vec3f Pb(pb);
        cv::Vec3f Pc(pc);
        cv::Vec3f Pe(pe);
        //-- Compute an orthonormal basis for the screen.
        cv::Vec3f Vr = Pb - Pa;
        Vr /= cv::norm(Vr);
        cv::Vec3f Vu = Pc - Pa;
        Vu /= cv::norm(Vu);
        cv::Vec3f Vn = Vr.cross(Vu);
        Vn /= cv::norm(Vn);
        //-- Compute the screen corner vectors.
        cv::Vec3f Va = Pa - Pe;
        cv::Vec3f Vb = Pb - Pe;
        cv::Vec3f Vc = Pc - Pe;
        //-- Find the distance from the eye to screen plane.
        float d = -Va.dot(Vn);
        //-- Find the extent of the perpendicular projection.
        float l = Va.dot(Vr) * near_ / d;
        float r = Vr.dot(Vb) * near_ / d;
        float b = Vu.dot(Va) * near_ / d;
        float t = Vu.dot(Vc) * near_ / d;
        
        //-- Load the perpendicular projection.
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glFrustum(l, r, b, t, near_, double(far_) + d);
        //-- Rotate the projection to be non-perpendicular.
        float M[16];
        memset(M, 0, 16 * sizeof(float));
        M[0] = Vr[0]; M[1] = Vu[0]; M[2] = Vn[0];
        M[4] = Vr[1]; M[5] = Vu[1]; M[6] = Vn[1];
        M[8] = Vr[2]; M[9] = Vu[2]; M[10] = Vn[2];
        M[15] = 1.0f;
        glMultMatrixf(M);
        //-- Move the apex of the frustum to the origin.
        glTranslatef(-pe[0], -pe[1], -pe[2]);
        glRotatef(180, 0, 1, 0);
        //-- Reset modelview matrix
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }
    else
    {
        //-- intrinsic camera params
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(60, (float)width / (float)height, 1, 250);
        //-- extrinsic camera params
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(glCamX, glCamY, glCamZ, 0, 0, 0, 0, 1, 0);
    }
}
cv::Mat detectEyes(cv::Mat image)
{
    // INIT
    std::vector<cv::Rect> faces;
    cv::Mat image_gray;
    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(image_gray, image_gray);


    // DETECT FACE
    //-- Find bigger face (opencv documentation)
    face_cascade.detectMultiScale(image_gray, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE | cv::CASCADE_FIND_BIGGEST_OBJECT, cv::Size(minFaceSize, minFaceSize));


    for (size_t i = 0; i < faces.size(); i++)
    {
        // DETECT EYES
        //-- points in pixel
        cv::Point leftEyePt(faces[i].x + faces[i].width * 0.30, faces[i].y + faces[i].height * 0.37);
        cv::Point rightEyePt(faces[i].x + faces[i].width * 0.70, faces[i].y + faces[i].height * 0.37);
        cv::Point eyeCenterPt(faces[i].x + faces[i].width * 0.5, leftEyePt.y);

        //-- normalize with webcam internal parameters
        GLdouble normRightEye = (rightEyePt.x - camWidth / 2) / f;
        GLdouble normLeftEye = (leftEyePt.x - camWidth / 2) / f;
        GLdouble normCenterX = (eyeCenterPt.x - camWidth / 2) / f;
        GLdouble normCenterY = (eyeCenterPt.y - camHeight / 2) / f;

        //-- get space coordinates
        float tempZ = eyesGap / (normRightEye - normLeftEye);
        float tempX = normCenterX * glCamZ;
        float tempY = -normCenterY * glCamZ;

        //-- update cam coordinates (smoothing)
        glCamX = (glCamX * 0.5) + (tempX * 0.5);
        glCamY = (glCamY * 0.5) + (tempY * 0.5);
        if (!lockZ) {
            k = (abs(tempZ - filteredVal) / eyesGap) / 3.5;
            if (k < 0.1) k = 0.1;
            if (k > 1) k = 1;
            //std::cout << k << std::endl;
            filteredVal += (tempZ - filteredVal) * k;
            glCamZ = filteredVal;
        }
        else
            glCamZ = lockedHeadDistance;
        

        // DISPLAY
        if (displayCam && displayDetection)
        {
            //-- face rectangle
            cv::rectangle(image, faces[i], 1234);

            //-- face lines
            cv::Point leftPt(faces[i].x, faces[i].y + faces[i].height * 0.37);
            cv::Point rightPt(faces[i].x + faces[i].width, faces[i].y + faces[i].height * 0.37);
            cv::Point topPt(faces[i].x + faces[i].width * 0.5, faces[i].y);
            cv::Point bottomPt(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height);
            cv::line(image, leftPt, rightPt, cv::Scalar(0, 0, 0), 1, 1, 0);
            cv::line(image, topPt, bottomPt, cv::Scalar(0, 0, 0), 1, 1, 0);

            //-- eyes circles
            cv::circle(image, rightEyePt, 0.06 * faces[i].width, cv::Scalar(255, 255, 255), 1, 8, 0);
            cv::circle(image, leftEyePt, 0.06 * faces[i].width, cv::Scalar(255, 255, 255), 1, 8, 0);

            //-- eyes line & center
            cv::line(image, leftEyePt, rightEyePt, cv::Scalar(0, 0, 255), 1, 1, 0);
            cv::circle(image, eyeCenterPt, 2, cv::Scalar(0, 0, 255), 3, 1, 0);
        }
    }
    return image;
}
void displayCamera(cv::Mat camImage)
{
    //-- Save matrix
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0.0, width, 0.0, height);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    /*
    //-- Display Coordinates
    if (displayDetection)
    {
        //-- Coord text
        std::stringstream sstm;
        sstm << "(x,y,z) = (" << (int)glCamX << "," << (int)glCamY << "," << (int)glCamZ << ")";
        std::string s = sstm.str();
        //std::cout<<s<<std::endl;

        //-- Display text
        glColor3f(1.0, 1.0, 1.0);
        glRasterPos2i(10, windowHeight - (camRatio * camImage.size().height) - 20);
        void* font = GLUT_BITMAP_9_BY_15;
        for (std::string::iterator i = s.begin(); i != s.end(); ++i)
        {
            char c = *i;
            glutBitmapCharacter(font, c);
        }
    }
    */
    //-- Display image
    glRasterPos2i(0, height - (camRatio * camImage.size().height));
    cv::flip(camImage, camImage, 0);
    cv::resize(camImage, camImage, cv::Size(camRatio * camWidth, camRatio * camHeight), 0, 0, cv::INTER_CUBIC);
    glDrawPixels(camImage.size().width, camImage.size().height, GL_BGR, GL_UNSIGNED_BYTE, camImage.ptr());

    //-- Load matrix
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

//model loading functions
static std::string GetBaseDir(const std::string& filepath) {
    if (filepath.find_last_of("/\\") != std::string::npos)
        return filepath.substr(0, filepath.find_last_of("/\\"));
    return "";
}
static bool FileExists(const std::string& abs_filename) {
    bool ret;
    FILE* fp = fopen(abs_filename.c_str(), "rb");
    if (fp) {
        ret = true;
        fclose(fp);
    }
    else {
        ret = false;
    }

    return ret;
}
static void CheckErrors(std::string desc) {
    GLenum e = glGetError();
    if (e != GL_NO_ERROR) {
        fprintf(stderr, "OpenGL error in \"%s\": %d (%d)\n", desc.c_str(), e, e);
        exit(20);
    }
}
static void CalcNormal(float N[3], float v0[3], float v1[3], float v2[3]) {
    float v10[3];
    v10[0] = v1[0] - v0[0];
    v10[1] = v1[1] - v0[1];
    v10[2] = v1[2] - v0[2];

    float v20[3];
    v20[0] = v2[0] - v0[0];
    v20[1] = v2[1] - v0[1];
    v20[2] = v2[2] - v0[2];

    N[0] = v10[1] * v20[2] - v10[2] * v20[1];
    N[1] = v10[2] * v20[0] - v10[0] * v20[2];
    N[2] = v10[0] * v20[1] - v10[1] * v20[0];

    float len2 = N[0] * N[0] + N[1] * N[1] + N[2] * N[2];
    if (len2 > 0.0f) {
        float len = sqrtf(len2);

        N[0] /= len;
        N[1] /= len;
        N[2] /= len;
    }
}
namespace  // Local utility functions
{
    struct vec3 {
        float v[3];
        vec3() {
            v[0] = 0.0f;
            v[1] = 0.0f;
            v[2] = 0.0f;
        }
    };

    void normalizeVector(vec3& v) {
        float len2 = v.v[0] * v.v[0] + v.v[1] * v.v[1] + v.v[2] * v.v[2];
        if (len2 > 0.0f) {
            float len = sqrtf(len2);

            v.v[0] /= len;
            v.v[1] /= len;
            v.v[2] /= len;
        }
    }

    // Check if `mesh_t` contains smoothing group id.
    bool hasSmoothingGroup(const tinyobj::shape_t& shape)
    {
        for (size_t i = 0; i < shape.mesh.smoothing_group_ids.size(); i++) {
            if (shape.mesh.smoothing_group_ids[i] > 0) {
                return true;
            }
        }
        return false;
    }

    void computeSmoothingNormals(const tinyobj::attrib_t& attrib, const tinyobj::shape_t& shape,
        std::map<int, vec3>& smoothVertexNormals) {
        smoothVertexNormals.clear();
        std::map<int, vec3>::iterator iter;

        for (size_t f = 0; f < shape.mesh.indices.size() / 3; f++) {
            // Get the three indexes of the face (all faces are triangular)
            tinyobj::index_t idx0 = shape.mesh.indices[3 * f + 0];
            tinyobj::index_t idx1 = shape.mesh.indices[3 * f + 1];
            tinyobj::index_t idx2 = shape.mesh.indices[3 * f + 2];

            // Get the three vertex indexes and coordinates
            int vi[3];      // indexes
            float v[3][3];  // coordinates

            for (int k = 0; k < 3; k++) {
                vi[0] = idx0.vertex_index;
                vi[1] = idx1.vertex_index;
                vi[2] = idx2.vertex_index;
                assert(vi[0] >= 0);
                assert(vi[1] >= 0);
                assert(vi[2] >= 0);

                v[0][k] = attrib.vertices[3 * vi[0] + k];
                v[1][k] = attrib.vertices[3 * vi[1] + k];
                v[2][k] = attrib.vertices[3 * vi[2] + k];
            }

            // Compute the normal of the face
            float normal[3];
            CalcNormal(normal, v[0], v[1], v[2]);

            // Add the normal to the three vertexes
            for (size_t i = 0; i < 3; ++i) {
                iter = smoothVertexNormals.find(vi[i]);
                if (iter != smoothVertexNormals.end()) {
                    // add
                    iter->second.v[0] += normal[0];
                    iter->second.v[1] += normal[1];
                    iter->second.v[2] += normal[2];
                }
                else {
                    smoothVertexNormals[vi[i]].v[0] = normal[0];
                    smoothVertexNormals[vi[i]].v[1] = normal[1];
                    smoothVertexNormals[vi[i]].v[2] = normal[2];
                }
            }

        }  // f

        // Normalize the normals, that is, make them unit vectors
        for (iter = smoothVertexNormals.begin(); iter != smoothVertexNormals.end();
            iter++) {
            normalizeVector(iter->second);
        }

    }  // computeSmoothingNormals
}  // namespace
static bool LoadObjAndConvert(float bmin[3], float bmax[3],
    std::vector<DrawObject>* drawObjects,
    std::vector<tinyobj::material_t>& materials,
    std::map<std::string, GLuint>& textures,
    const char* filename) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;

    std::string base_dir = GetBaseDir(filename);
    if (base_dir.empty()) {
        base_dir = ".";
    }
#ifdef _WIN32
    base_dir += "\\";
#else
    base_dir += "/";
#endif

    std::string warn;
    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename,
        base_dir.c_str());
    if (!warn.empty()) {
        std::cout << "WARN: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        std::cerr << "Failed to load " << filename << std::endl;
        return false;
    }

    printf("# of vertices  = %d\n", (int)(attrib.vertices.size()) / 3);
    printf("# of normals   = %d\n", (int)(attrib.normals.size()) / 3);
    printf("# of texcoords = %d\n", (int)(attrib.texcoords.size()) / 2);
    printf("# of materials = %d\n", (int)materials.size());
    printf("# of shapes    = %d\n", (int)shapes.size());

    // Append `default` material
    materials.push_back(tinyobj::material_t());

    for (size_t i = 0; i < materials.size(); i++) {
        printf("material[%d].diffuse_texname = %s\n", int(i),
            materials[i].diffuse_texname.c_str());
    }

    // Load diffuse textures
    {
        for (size_t m = 0; m < materials.size(); m++) {
            tinyobj::material_t* mp = &materials[m];

            if (mp->diffuse_texname.length() > 0) {
                // Only load the texture if it is not already loaded
                if (textures.find(mp->diffuse_texname) == textures.end()) {
                    GLuint texture_id;
                    int w, h;
                    int comp;

                    std::string texture_filename = mp->diffuse_texname;
                    if (!FileExists(texture_filename)) {
                        // Append base dir.
                        texture_filename = base_dir + mp->diffuse_texname;
                        if (!FileExists(texture_filename)) {
                            std::cerr << "Unable to find file: " << mp->diffuse_texname
                                << std::endl;
                            exit(1);
                        }
                    }

                    unsigned char* image =
                        stbi_load(texture_filename.c_str(), &w, &h, &comp, STBI_default);
                    if (!image) {
                        std::cerr << "Unable to load texture: " << texture_filename
                            << std::endl;
                        exit(1);
                    }
                    std::cout << "Loaded texture: " << texture_filename << ", w = " << w
                        << ", h = " << h << ", comp = " << comp << std::endl;

                    glGenTextures(1, &texture_id);
                    glBindTexture(GL_TEXTURE_2D, texture_id);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                    if (comp == 3) {
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB,
                            GL_UNSIGNED_BYTE, image);
                    }
                    else if (comp == 4) {
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA,
                            GL_UNSIGNED_BYTE, image);
                    }
                    else {
                        assert(0);  // TODO
                    }
                    glBindTexture(GL_TEXTURE_2D, 0);
                    stbi_image_free(image);
                    textures.insert(std::make_pair(mp->diffuse_texname, texture_id));
                }
            }
        }
    }

    bmin[0] = bmin[1] = bmin[2] = std::numeric_limits<float>::max();
    bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits<float>::max();

    {
        for (size_t s = 0; s < shapes.size(); s++) {
            DrawObject o;
            std::vector<float> buffer;  // pos(3float), normal(3float), color(3float)

            // Check for smoothing group and compute smoothing normals
            std::map<int, vec3> smoothVertexNormals;
            if (hasSmoothingGroup(shapes[s]) > 0) {
                std::cout << "Compute smoothingNormal for shape [" << s << "]" << std::endl;
                computeSmoothingNormals(attrib, shapes[s], smoothVertexNormals);
            }

            for (size_t f = 0; f < shapes[s].mesh.indices.size() / 3; f++) {
                tinyobj::index_t idx0 = shapes[s].mesh.indices[3 * f + 0];
                tinyobj::index_t idx1 = shapes[s].mesh.indices[3 * f + 1];
                tinyobj::index_t idx2 = shapes[s].mesh.indices[3 * f + 2];

                int current_material_id = shapes[s].mesh.material_ids[f];

                if ((current_material_id < 0) ||
                    (current_material_id >= static_cast<int>(materials.size()))) {
                    // Invaid material ID. Use default material.
                    current_material_id =
                        materials.size() -
                        1;  // Default material is added to the last item in `materials`.
                }
                // if (current_material_id >= materials.size()) {
                //    std::cerr << "Invalid material index: " << current_material_id <<
                //    std::endl;
                //}
                //
                float diffuse[3];
                for (size_t i = 0; i < 3; i++) {
                    diffuse[i] = materials[current_material_id].diffuse[i];
                }
                float tc[3][2];
                if (attrib.texcoords.size() > 0) {
                    if ((idx0.texcoord_index < 0) || (idx1.texcoord_index < 0) ||
                        (idx2.texcoord_index < 0)) {
                        // face does not contain valid uv index.
                        tc[0][0] = 0.0f;
                        tc[0][1] = 0.0f;
                        tc[1][0] = 0.0f;
                        tc[1][1] = 0.0f;
                        tc[2][0] = 0.0f;
                        tc[2][1] = 0.0f;
                    }
                    else {
                        assert(attrib.texcoords.size() >
                            size_t(2 * idx0.texcoord_index + 1));
                        assert(attrib.texcoords.size() >
                            size_t(2 * idx1.texcoord_index + 1));
                        assert(attrib.texcoords.size() >
                            size_t(2 * idx2.texcoord_index + 1));

                        // Flip Y coord.
                        tc[0][0] = attrib.texcoords[2 * idx0.texcoord_index];
                        tc[0][1] = 1.0f - attrib.texcoords[2 * idx0.texcoord_index + 1];
                        tc[1][0] = attrib.texcoords[2 * idx1.texcoord_index];
                        tc[1][1] = 1.0f - attrib.texcoords[2 * idx1.texcoord_index + 1];
                        tc[2][0] = attrib.texcoords[2 * idx2.texcoord_index];
                        tc[2][1] = 1.0f - attrib.texcoords[2 * idx2.texcoord_index + 1];
                    }
                }
                else {
                    tc[0][0] = 0.0f;
                    tc[0][1] = 0.0f;
                    tc[1][0] = 0.0f;
                    tc[1][1] = 0.0f;
                    tc[2][0] = 0.0f;
                    tc[2][1] = 0.0f;
                }

                float v[3][3];
                for (int k = 0; k < 3; k++) {
                    int f0 = idx0.vertex_index;
                    int f1 = idx1.vertex_index;
                    int f2 = idx2.vertex_index;
                    assert(f0 >= 0);
                    assert(f1 >= 0);
                    assert(f2 >= 0);

                    v[0][k] = attrib.vertices[3 * f0 + k];
                    v[1][k] = attrib.vertices[3 * f1 + k];
                    v[2][k] = attrib.vertices[3 * f2 + k];
                    bmin[k] = std::min(v[0][k], bmin[k]);
                    bmin[k] = std::min(v[1][k], bmin[k]);
                    bmin[k] = std::min(v[2][k], bmin[k]);
                    bmax[k] = std::max(v[0][k], bmax[k]);
                    bmax[k] = std::max(v[1][k], bmax[k]);
                    bmax[k] = std::max(v[2][k], bmax[k]);
                }

                float n[3][3];
                {
                    bool invalid_normal_index = false;
                    if (attrib.normals.size() > 0) {
                        int nf0 = idx0.normal_index;
                        int nf1 = idx1.normal_index;
                        int nf2 = idx2.normal_index;

                        if ((nf0 < 0) || (nf1 < 0) || (nf2 < 0)) {
                            // normal index is missing from this face.
                            invalid_normal_index = true;
                        }
                        else {
                            for (int k = 0; k < 3; k++) {
                                assert(size_t(3 * nf0 + k) < attrib.normals.size());
                                assert(size_t(3 * nf1 + k) < attrib.normals.size());
                                assert(size_t(3 * nf2 + k) < attrib.normals.size());
                                n[0][k] = attrib.normals[3 * nf0 + k];
                                n[1][k] = attrib.normals[3 * nf1 + k];
                                n[2][k] = attrib.normals[3 * nf2 + k];
                            }
                        }
                    }
                    else {
                        invalid_normal_index = true;
                    }

                    if (invalid_normal_index && !smoothVertexNormals.empty()) {
                        // Use smoothing normals
                        int f0 = idx0.vertex_index;
                        int f1 = idx1.vertex_index;
                        int f2 = idx2.vertex_index;

                        if (f0 >= 0 && f1 >= 0 && f2 >= 0) {
                            n[0][0] = smoothVertexNormals[f0].v[0];
                            n[0][1] = smoothVertexNormals[f0].v[1];
                            n[0][2] = smoothVertexNormals[f0].v[2];

                            n[1][0] = smoothVertexNormals[f1].v[0];
                            n[1][1] = smoothVertexNormals[f1].v[1];
                            n[1][2] = smoothVertexNormals[f1].v[2];

                            n[2][0] = smoothVertexNormals[f2].v[0];
                            n[2][1] = smoothVertexNormals[f2].v[1];
                            n[2][2] = smoothVertexNormals[f2].v[2];

                            invalid_normal_index = false;
                        }
                    }

                    if (invalid_normal_index) {
                        // compute geometric normal
                        CalcNormal(n[0], v[0], v[1], v[2]);
                        n[1][0] = n[0][0];
                        n[1][1] = n[0][1];
                        n[1][2] = n[0][2];
                        n[2][0] = n[0][0];
                        n[2][1] = n[0][1];
                        n[2][2] = n[0][2];
                    }
                }

                for (int k = 0; k < 3; k++) {
                    buffer.push_back(v[k][0]);
                    buffer.push_back(v[k][1]);
                    buffer.push_back(v[k][2]);
                    buffer.push_back(n[k][0]);
                    buffer.push_back(n[k][1]);
                    buffer.push_back(n[k][2]);
                    // Combine normal and diffuse to get color.
                    float normal_factor = 0.2;
                    float diffuse_factor = 1 - normal_factor;
                    float c[3] = { n[k][0] * normal_factor + diffuse[0] * diffuse_factor,
                                  n[k][1] * normal_factor + diffuse[1] * diffuse_factor,
                                  n[k][2] * normal_factor + diffuse[2] * diffuse_factor };
                    float len2 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
                    if (len2 > 0.0f) {
                        float len = sqrtf(len2);

                        c[0] /= len;
                        c[1] /= len;
                        c[2] /= len;
                    }
                    buffer.push_back(c[0] * 0.5 + 0.5);
                    buffer.push_back(c[1] * 0.5 + 0.5);
                    buffer.push_back(c[2] * 0.5 + 0.5);

                    buffer.push_back(tc[k][0]);
                    buffer.push_back(tc[k][1]);
                }
            }

            o.vb_id = 0;
            o.numTriangles = 0;

            // OpenGL viewer does not support texturing with per-face material.
            if (shapes[s].mesh.material_ids.size() > 0 &&
                shapes[s].mesh.material_ids.size() > s) {
                o.material_id = shapes[s].mesh.material_ids[0];  // use the material ID
                                                                 // of the first face.
            }
            else {
                o.material_id = materials.size() - 1;  // = ID for default material.
            }
            printf("shape[%d] material_id %d\n", int(s), int(o.material_id));

            if (buffer.size() > 0) {
                glGenBuffers(1, &o.vb_id);
                glBindBuffer(GL_ARRAY_BUFFER, o.vb_id);
                glBufferData(GL_ARRAY_BUFFER, buffer.size() * sizeof(float),
                    &buffer.at(0), GL_STATIC_DRAW);
                o.numTriangles = buffer.size() / (3 + 3 + 3 + 2) /
                    3;  // 3:vtx, 3:normal, 3:col, 2:texcoord

                printf("shape[%d] # of triangles = %d\n", static_cast<int>(s),
                    o.numTriangles);
            }

            drawObjects->push_back(o);
        }
    }

    printf("bmin = %f, %f, %f\n", bmin[0], bmin[1], bmin[2]);
    printf("bmax = %f, %f, %f\n", bmax[0], bmax[1], bmax[2]);

    return true;
}

//callback functions
static void reshapeFunc(GLFWwindow* window, int w, int h) {
    int fb_w, fb_h;
    // Get actual framebuffer size.
    glfwGetFramebufferSize(window, &fb_w, &fb_h);

    glViewport(0, 0, fb_w, fb_h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (float)w / (float)h, 0.01f, 100.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    cx = pixelToCm(w);
    cy = pixelToCm(h);

    width = w;
    height = h;
}
static void keyboardFunc(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)window;
    (void)scancode;
    (void)mods;
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        // Close window
        if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE)
            glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

//display function
static void Draw(const std::vector<DrawObject>& drawObjects,
    std::vector<tinyobj::material_t>& materials,
    std::map<std::string, GLuint>& textures) {
    glPolygonMode(GL_FRONT, GL_FILL);
    glPolygonMode(GL_BACK, GL_FILL);

    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0, 1.0);
    GLsizei stride = (3 + 3 + 3 + 2) * sizeof(float);
    for (size_t i = 0; i < drawObjects.size(); i++) {
        DrawObject o = drawObjects[i];
        if (o.vb_id < 1) {
            continue;
        }

        glBindBuffer(GL_ARRAY_BUFFER, o.vb_id);
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);

        glBindTexture(GL_TEXTURE_2D, 0);
        if ((o.material_id < materials.size())) {
            std::string diffuse_texname = materials[o.material_id].diffuse_texname;
            if (textures.find(diffuse_texname) != textures.end()) {
                glBindTexture(GL_TEXTURE_2D, textures[diffuse_texname]);
                if (!textureFiltering) {
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                }
            }
        }
        glVertexPointer(3, GL_FLOAT, stride, (const void*)0);
        glNormalPointer(GL_FLOAT, stride, (const void*)(sizeof(float) * 3));
        glColorPointer(3, GL_FLOAT, stride, (const void*)(sizeof(float) * 6));
        glTexCoordPointer(2, GL_FLOAT, stride, (const void*)(sizeof(float) * 9));

        glDrawArrays(GL_TRIANGLES, 0, 3 * o.numTriangles);
        CheckErrors("drawarrays");
        glBindTexture(GL_TEXTURE_2D, 0);
    }


    if(drawWireframe){
        glDisable(GL_POLYGON_OFFSET_FILL);
        glPolygonMode(GL_FRONT, GL_LINE);
        glPolygonMode(GL_BACK, GL_LINE);

        glColor3f(0.0f, 0.0f, 0.4f);
        for (size_t i = 0; i < drawObjects.size(); i++) {
            DrawObject o = drawObjects[i];
            if (o.vb_id < 1) {
                continue;
            }

            glBindBuffer(GL_ARRAY_BUFFER, o.vb_id);
            glEnableClientState(GL_VERTEX_ARRAY);
            glEnableClientState(GL_NORMAL_ARRAY);
            glDisableClientState(GL_COLOR_ARRAY);
            glDisableClientState(GL_TEXTURE_COORD_ARRAY);
            glVertexPointer(3, GL_FLOAT, stride, (const void*)0);
            glNormalPointer(GL_FLOAT, stride, (const void*)(sizeof(float) * 3));
            glColorPointer(3, GL_FLOAT, stride, (const void*)(sizeof(float) * 6));
            glTexCoordPointer(2, GL_FLOAT, stride, (const void*)(sizeof(float) * 9));

            glDrawArrays(GL_TRIANGLES, 0, 3 * o.numTriangles);
            CheckErrors("drawarrays");
        }
    }
}

int main(int argc, char** argv) {

    //HANDLING WALLPAPER ENGINE ARGUMENTS
    if (argc >= 10 && argv[8] == "-wpe_h") {
        try {
            height = std::stoi(argv[9]);
        }
        catch (...) {
            ;
        }
    }
    if (argc >= 12 && argv[10] == "-wpe_w") {
        try {
            width = std::stoi(argv[11]);
        }
        catch (...) {
            ;
        }
    }

    // READ CONFIG
    if (readConfig(configPath))
        std::cout << "Loaded successful.\n";
    else
        std::cout << "Error. Continuing with default settings.\n";

    // OPENCV INIT
    if (!face_cascade.load(face_cascade_name))
    {
        std::cout << "-- (!) ERROR loading 'haarcascade_frontalface_alt.xml'\n";
        std::cout << "Please edit 'face_cascade_name' path in main.cpp:22 and recompile the project.\n";
        return -1;
    };

    // VIDEO CAPTURE
    int camerasAvalible = countCameras();
    if (camerasAvalible-1 < videoCaptureDevice)
        videoCaptureDevice = camerasAvalible-1;
    capture = new cv::VideoCapture(videoCaptureDevice);
    if (capture == NULL || !capture->isOpened()) {
        if (camerasAvalible == 0)
            std::cout << "No cameras avalible\n";
        std::cout << "Could not start video capture\n";
        return 1;
    }

    // CAMERA IMAGE DIMENSIONS
    camWidth = (int)capture->get(cv::CAP_PROP_FRAME_WIDTH);
    camHeight = (int)capture->get(cv::CAP_PROP_FRAME_HEIGHT);
    cx = pixelToCm(width);
    cy = pixelToCm(height);

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW." << std::endl;
        return -1;
    }

    window = glfwCreateWindow(width, height, ("ArWallpaper " + version).c_str(), NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to open GLFW window. " << std::endl;
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Callback
    glfwSetWindowSizeCallback(window, reshapeFunc);
    glfwSetKeyCallback(window, keyboardFunc);

    glewExperimental = true;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW." << std::endl;
        return -1;
    }
    //glutInit(&argc, argv);
    reshapeFunc(window, width, height);

    float bmin[3], bmax[3];
    std::vector<tinyobj::material_t> materials;
    std::map<std::string, GLuint> textures;
    if (false == LoadObjAndConvert(bmin, bmax, &gDrawObjects, materials, textures,
        modelPath.c_str())) {
        return -1;
    }

    float maxExtent = 0.5f * (bmax[0] - bmin[0]);
    if (maxExtent < 0.5f * (bmax[1] - bmin[1])) {
        maxExtent = 0.5f * (bmax[1] - bmin[1]);
    }
    if (maxExtent < 0.5f * (bmax[2] - bmin[2])) {
        maxExtent = 0.5f * (bmax[2] - bmin[2]);
    }
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    //
    glAlphaFunc(GL_GREATER, 0.5);
    glEnable(GL_ALPHA_TEST);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //glEnable(GL_BLEND);
    //
    while (glfwWindowShouldClose(window) == GL_FALSE) {
        glfwPollEvents();
        glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        (*capture) >> frame;

        cv::Mat tempimage;
        if (invertCam) cv::flip(frame, tempimage, 0);
        else cv::flip(frame, tempimage, 1);
        //-- detect eyes
        tempimage = detectEyes(tempimage);
        setGlCamera();

        Draw(gDrawObjects, materials, textures);
        //if (displayCam) displayCamera(tempimage);

        glfwSwapBuffers(window);
    }

    glfwTerminate();
}
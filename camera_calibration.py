#Summary- This code performs camera calibration for single camera and stereo camera and triangulates points from 2D to 3D with undistortion code
#Author - Ajay
#Created: 06/22/2023
#Last updated: 10/20/2023

#import statements
import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt


class Camera_Calibration:
    #folders containing images for intrinsic calibration
    def __init__(self,cam0_img_folder,cam1_img_folder):
        self.cam0_img_folder=cam0_img_folder
        self.cam1_img_folder= cam1_img_folder
        self.h=0
        self.w=0
        
    #this function takes in a folder and extracts all the images in the folder and returns it as a list
    def get_image_files(self,folder_name):

        image_files= list()
        timestamps = list()
        if os.path.exists(folder_name):
            for path, names, files in os.walk(folder_name):
                for f in files:
                    if os.path.splitext(f)[1] in ['.png', '.tif', '.jpg', '.bmp']:
                        image_files.append(os.path.join(path,f))
                        f_name= os.path.splitext(f)[0]

                        t_stamp= f_name[-5:][f_name[-5:].find('_')+1:]
                        
                        timestamps.append(t_stamp)  

                break

        sort_list = sorted(zip(timestamps, image_files))
        image_files = [file[1] for file in sort_list]

        return image_files
    

    def detect_checker_corners(self, image_files, verify_corners=False, custom_pts_path=None):

        #define checkerboard
        CHECKERBOARD = (6, 6)
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        world_scaling = 4 #4cm

        threedpoints = []
        twodpoints = []

        #coordinates of squares in the checkerboard world space
        objectp3d = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
        objectp3d[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
        
        objectp3d = world_scaling* objectp3d
        
        #find checkerboard corners
        for filename in image_files:
            image = cv2.imread(filename)

            if self.h == 0:
                self.h = image.shape[0]
                self.w = image.shape[1]
                print("Image size: ", (self.h,self.w))
            #size 1280x720
            # image = cv2.resize(image, (720, 720), interpolation = cv2.INTER_AREA)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH+ cv2.CALIB_CB_EXHAUSTIVE +cv2.CALIB_CB_NORMALIZE_IMAGE,)

            if ret == True:

                # termination criteria
                corners = cv2.cornerSubPix(gray,corners, (3,3), (-1,-1), criteria)

                diff = corners[0][0][0] - corners[CHECKERBOARD[0]-1][0][0]
                if diff > 0:
                    corners = corners[::-1]

                image = cv2.drawChessboardCorners(image,CHECKERBOARD,corners, ret)
                # print(filename)

                if verify_corners == True:
                    cv2.imshow('img', image)
                    if cv2.waitKey(0) == ord('s'):
                        print("Saving points for: ", filename)
                        threedpoints.append(objectp3d)
                        twodpoints.append(corners)

                    elif cv2.waitKey(0) == ord('d'):
                        print("Skipping image file: ",filename) 

                else:    
                    threedpoints.append(objectp3d)
                    twodpoints.append(corners) 
                    
            if ret == False:
                print("Failed image name:", filename)
                # print(use_custom_pts)

                if custom_pts_path != None:
                    print("Using custom marked points...")
                    ####### code to use custom marked points ########
                    #load pickled points list
                    with open(custom_pts_path, 'rb') as file:
                        points_dict= pickle.load(file)

                    img_names = list(points_dict.keys())

                    for key_name in img_names:
                        
                        #get the image names
                        rev_filename = filename[::-1]
                        img_filename = rev_filename[:rev_filename.find("\\")][::-1]                      

                        if key_name == img_filename:

                            points_arr = np.array([points_dict[img_filename]]).reshape((36,1,2)) #shape: (36,1,2)
                            float_pts_arr=np.zeros((36,1,2), dtype="float32")

                            for i in range(len(points_arr)):

                                float_pts_arr[i][0][0] = float(points_arr[i][0][0])
                                float_pts_arr[i][0][1] = float(points_arr[i][0][1])

                            twodpoints.append(float_pts_arr)
                            threedpoints.append(objectp3d)

                else:
                    print("No custom points have been provided!!!")

        cv2.destroyAllWindows()

        print("Number of images with marked corners: ", len(twodpoints))

        return threedpoints, twodpoints, gray
    
    def check_undistortion(self, cam_matrix, dist, distorted_imgs, path_to_save_undistorted, save_undistorted_img=True):

        # h,w = test_imgs[0].shape[:2]
        # print("Number of images: ",len(distorted_imgs))

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam_matrix, dist, (self.w,self.h), 1, (self.w,self.h))

        # map0_x, map0_y = cv2.initUndistortRectifyMap(cam_matrix, dist, None, newcameramtx, (self.h,self.w), cv2.CV_16SC2)

        folder_name= '/undistorted_imgs'
        for img_name in distorted_imgs:

            img= cv2.imread(img_name)

            # im_remapped_0 = cv2.remap(img, map0_x, map0_y, cv2.INTER_LANCZOS4)
            im_remapped_0 = cv2.undistort(img, cam_matrix, dist, None, newcameramtx)

            # crop the image
            # x, y, w, h = roi
            # im_remapped_0 = im_remapped_0[y:y+h, x:x+w]

            if save_undistorted_img == True:
                print("Saving undistorted images", im_remapped_0.shape)

                if os.path.exists(path_to_save_undistorted + folder_name) != True:
                    
                    # print(path_to_save_undistorted + folder_name,os.path.exists(path_to_save_undistorted + folder_name))
                    os.mkdir(path_to_save_undistorted + folder_name)

                uscore_id= img_name.find('_') 

                op_file_name = path_to_save_undistorted  + folder_name + img_name[uscore_id-9:][:-4] + "_undistorted.png"

                cv2.imwrite(op_file_name, im_remapped_0)

            # cv2.imshow('img', im_remapped_0)
            # cv2.waitKey(0)

        # cv2.destroyAllWindows()



    def calibrate_one_camera(self, threedpoints, twodpoints, gray, check_undistortion=True):

        # print("Calibrating........")
        print(gray.shape[::-1])
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, gray.shape[::-1], None, None)
        # matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (self.w,self.h), 1, (self.w,self.h))
    
        print("Projection matrix")
        print(matrix)
        print("Distortion")
        print(distortion)

        # objpoints=threedpoints
        # imgpoints=twodpoints

        mean_error = 0

        for i in range(len(threedpoints)):
            imgpoints2, _ = cv2.projectPoints(threedpoints[i], r_vecs[i], t_vecs[i], matrix, distortion)
            error = cv2.norm(twodpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error

        repr_error= mean_error/len(threedpoints)
        print("Error: ",repr_error)
        print("         ")

        return ret, matrix, distortion, r_vecs, t_vecs, repr_error
    

    def dist_3d_pts(self, p0,p1):

        return np.linalg.norm(p0 - p1)
    

    def stereo_calibrate(self,image_folder0,image_folder1, output_file_name='calibration_result.pickle', check_undistortion=False, custom_pts_filename = None):
        print(" ")
        print("############  Mono Calibration   #############")
        print(" ")

        #cam0
        # folder_name='dataset/cam_day6/intrinsic/cam0'
        image_files= self.get_image_files(self.cam0_img_folder)
        

        threedpoints, twodpoints, gray = self.detect_checker_corners(image_files)

        ret, matrix_0, distortion_0, r_vecs_0, t_vecs_0, repr_error_0 = self.calibrate_one_camera(threedpoints, twodpoints, gray)
        distortion_0[0][-3:] = np.array([0.,0.,0.])
        # distortion_0 = np.array([-0.37787529,  0.35057683,  0.00473403,  0.00141145, -0.24608929]) #using custom distortion that worked on another day's videos
    
        if check_undistortion == True:
            self.check_undistortion(cam_matrix=matrix_0, dist=distortion_0, distorted_imgs=image_files, path_to_save_undistorted=self.cam0_img_folder ,save_undistorted_img=True)

        #cam1
        # folder_name='dataset/cam_day6/intrinsic/cam1'
        image_files= self.get_image_files(self.cam1_img_folder)

        threedpoints, twodpoints, gray = self.detect_checker_corners(image_files)

        ret, matrix_1, distortion_1, r_vecs_1, t_vecs_1, repr_error_1 = self.calibrate_one_camera(threedpoints, twodpoints, gray)
        distortion_1[0][-3:] = np.array([0.,0.,0.])
        # distortion_1 = np.array([-0.37787529,  0.35057683,  0.00473403,  0.00141145, -0.24608929]) #using custom distortion that worked on another day's videos

        if check_undistortion == True:
            self.check_undistortion(cam_matrix=matrix_1, dist=distortion_1, distorted_imgs=image_files, path_to_save_undistorted=self.cam1_img_folder,save_undistorted_img=True)

        #stereo
        criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # image_file0= ['dataset/cam_day6/extrinsic/sync_dets/cam0/21122010_083501_IMAQdxcam0.png']
        # image_file1= ['dataset/cam_day6/extrinsic/sync_dets/cam1/21122010_083501_IMAQdxcam1.png']

        image_files_0= self.get_image_files(folder_name=image_folder0)
        image_files_1= self.get_image_files(folder_name=image_folder1)
        # print(len(image_files_0), len(image_files_1))

        if custom_pts_filename != None:
            custom_pts_path = image_folder0+'/'+custom_pts_filename
        else:
            custom_pts_path = None

        threedpoints, twodpoints, gray = self.detect_checker_corners(image_files_0, custom_pts_path = custom_pts_path)
        
        img_points_cam0 = twodpoints
        obj_points = threedpoints

        # objpoints= threedpoints20110120-tm3-ug-ur
        # imgpoints_cam0= twodpoints

        if custom_pts_filename != None:
            custom_pts_path = image_folder1+'/'+custom_pts_filename

        threedpoints, twodpoints, gray = self.detect_checker_corners(image_files_1, custom_pts_path = custom_pts_path)

        img_points_cam1 = twodpoints

        # print("Number of image points in each cam: ",len(img_points_cam0[0]), len(img_points_cam1[0]))

        assert len(img_points_cam0) == len(img_points_cam1)

        img_points_cam0 = np.array(img_points_cam0)
        img_points_cam1 = np.array(img_points_cam1)
        obj_points = np.array(obj_points)

        print(" ")
        print("############  Stereo Calibration   #############")
        print(" ")

        # stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC

        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
 
        ret, CM1, dist0, CM2, dist1, R, T, E, F = cv2.stereoCalibrate(obj_points, img_points_cam0, img_points_cam1, matrix_0, distortion_0, matrix_1, distortion_1, (self.w, self.h), criteria = criteria, flags = flags)

        print("Stereo reprojection error: ", ret)
        print(" ")

        # R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(matrix_0, dist1, matrix_1, dist2, (self.w,self.h), R, T)
        # P1= P1[:,0:3]
        # P2= P2[:,0:3]

        # matrix_0 = P1
        # matrix_1 = P2
        #Save output
        calib_result={
            'camera_matrix_c0': matrix_0,
            'distortion_c0':dist0,
            'camera_matrix_c1': matrix_1,
            'distortion_c1':dist1,
            'c0_R_c1':R,
            'c0_t_c1':T,
            'essential_matrix':E,
            'fundamental_matrix':F,
            'stereo_repr_error':ret,
            'cam0_repr_error':repr_error_0,
            'cam1_repr_error':repr_error_1
            }
        print("Rotation:", R)
        print("Translation:",T)
        with open(output_file_name, 'wb') as file:
            pickle.dump(calib_result, file)

        print("Calibration output stored to ",output_file_name )


        #undistortion
        if check_undistortion == True:
            self.check_undistortion(cam_matrix=matrix_0, dist=dist0, distorted_imgs=image_files_0, path_to_save_undistorted=image_folder0,save_undistorted_img=True)

            self.check_undistortion(cam_matrix=matrix_1, dist=dist1, distorted_imgs=image_files_1, path_to_save_undistorted=image_folder1,save_undistorted_img=True)



    def get_projection_mat(self,matrix_0, matrix_1, R,T):
        #RT matrix for Cam0 is identity.
        RT0 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
        P0 = matrix_0 @ RT0 #projection matrix for Cam0
        
        #RT matrix for Cam1 is the R and T obtained from stereo calibration.
        RT1 = np.concatenate([R, T], axis = -1)
        P1 = matrix_1 @ RT1 #projection matrix for Cam1

        return P0, P1

    def DLT(self, P1, P2, point1, point2):
    
        A = [point1[1]*P1[2,:] - P1[1,:],
            P1[0,:] - point1[0]*P1[2,:],
            point2[1]*P2[2,:] - P2[1,:],
            P2[0,:] - point2[0]*P2[2,:]
            ]
        A = np.array(A).reshape((4,4))
        #print('A: ')
        #print(A)
    
        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices = False)
    
        return Vh[3,0:3]/Vh[3,3]
    
    # def dist_3d_pts(self,p0,p1):

    #     return np.linalg.norm(p0 - p1)

    #function to undistort an image
    # def undistort_img(self, projection_mtx, dist, img):
    #     #calculate new 

    def cv2_triangulate_pts(self, projection_mat_0, projection_mat_1, img_pts_0, img_pts_1):
        cv2_pts_3d= cv2.triangulatePoints(projection_mat_0, projection_mat_1, img_pts_0, img_pts_1)

        return cv2_pts_3d

    def triangulate_pts(self, projection_mat_0, projection_mat_1, img_pts_0, img_pts_1, frame_0=None, frame_1=None, show_triangulation_result=True, verbose= False):

        if verbose == True:
            print(" ")
            print("############  Triangulation   #############")
            print(" ")

            print("Number of points to triangluate: ", len(img_pts_0), len(img_pts_1))

        if frame_0 is not None:
            plt.imshow(frame_0[:,:,[2,1,0]])
            plt.scatter(img_pts_0[:,0], img_pts_0[:,1])
            plt.show()
        if frame_1 is not None:
            plt.imshow(frame_1[:,:,[2,1,0]])
            plt.scatter(img_pts_1[:,0], img_pts_1[:,1])
            plt.show()


        p3ds = []
        for uv0, uv1 in zip(img_pts_0, img_pts_1):
            _p3d = self.DLT(projection_mat_0, projection_mat_1, uv0, uv1)
            p3ds.append(_p3d)
        p3ds = np.array(p3ds)

        
        # print(p3ds)
        dist3d_list=[]
        for p in range(len(p3ds)-1):
            p0= p3ds[p][0:3]
            p1= p3ds[p+1][0:3]
            # print("P0,P1:", p0,p1)

            dist_3d=self.dist_3d_pts(p0,p1)
            dist3d_list.append(dist_3d)
            

        if verbose == True:
            print("3D points.........")
            print("   ")
            print("Distance between each points in XYZ")

            for pt in dist3d_list:
                print(pt)



        if show_triangulation_result == True:
            # from mpl_toolkits.mplot3d import Axes3D
            plt.style.use('seaborn-v0_8-notebook')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            x= p3ds[:,0]
            y= p3ds[:,1]
            z= p3ds[:,2]

            ax.scatter(x, y, z)

            plt.show()

            print("Success!!!")

        return p3ds, dist3d_list



############################   TESTING CALIBRATION MAIN CODE    #########################

def test_calibration():

    calib_cam = Camera_Calibration(cam0_img_folder='20110110-tm1-ub-maca/Calibration/Intrinseque/cam0',cam1_img_folder='20110110-tm1-ub-maca/Calibration/Intrinseque/cam1')

    calib_cam.stereo_calibrate(image_folder0='20110110-tm1-ub-maca/Calibration/Extrinseque/cam0', image_folder1='20110110-tm1-ub-maca/Calibration/Extrinseque/cam1', check_undistortion=False, custom_pts_filename= "manual_pts_data.pkl" )


    #test images- cam0 and cam1
    dist_test_img0='20101204-tm1-rb/Calibration/extrinsic/test/04122010_092624_IMAQdxcam0.png'
    dist_test_img1='20101204-tm1-rb/Calibration/extrinsic/test/04122010_092624_IMAQdxcam1.png'

    # print(frame_0.shape, frame_1.shape)

    #undistort test images
    """

    #load pickled calibration result
    with open('calibration_result.pickle', 'rb') as file:
        calib_result= pickle.load(file)
        matrix_0 = calib_result['camera_matrix_c0']
        matrix_1 = calib_result['camera_matrix_c1']
        dist0 = calib_result['distortion_c0']
        dist1 = calib_result['distortion_c1']


    calib_cam.check_undistortion(cam_matrix=matrix_0, dist=dist0, distorted_imgs=[dist_test_img0], path_to_save_undistorted='20101204-tm1-rb/Calibration/extrinsic/test/',save_undistorted_img=True)
    calib_cam.check_undistortion(cam_matrix=matrix_1, dist=dist1, distorted_imgs=[dist_test_img1], path_to_save_undistorted='20101204-tm1-rb/Calibration/extrinsic/test/',save_undistorted_img=True)

    #using undistorted images
    test_img0='20101204-tm1-rb/Calibration/extrinsic/test/undistorted_imgs/04122010_092624_IMAQdxcam0_undistorted.png'
    test_img1='20101204-tm1-rb/Calibration/extrinsic/test/undistorted_imgs/04122010_092624_IMAQdxcam1_undistorted.png'

    frame_0=cv2.imread(test_img0)  
    frame_1=cv2.imread(test_img1)

    #from manual_pts_data.pkl
    #load pickled points
    with open('20101204-tm1-rb/Calibration/extrinsic/test/undistorted_imgs/manual_pts_data.pkl', 'rb') as file: manual_pts_cam0= pickle.load(file)
    img_names = list(manual_pts_cam0.keys())
    twodpoints_cam0 = manual_pts_cam0[img_names[0]]

    # twodpoints_1 = manual_pts[img_names[1]]

    with open('20101204-tm1-rb/Calibration/extrinsic/test/undistorted_imgs/manual_pts_data.pkl', 'rb') as file: manual_pts_cam1= pickle.load(file)
    img_names = list(manual_pts_cam1.keys())
    twodpoints_cam1 = manual_pts_cam1[img_names[1]]

    """

    #triangulate without undistortion
    cam0_img_files = calib_cam.get_image_files("20110110-tm1-ub-maca/Calibration/Extrinseque/cam0")[0]
    cam1_img_files = calib_cam.get_image_files("20110110-tm1-ub-maca/Calibration/Extrinseque/cam1")[0]

    frame_0 = cv2.imread(cam0_img_files)
    frame_1 = cv2.imread(cam1_img_files)

    threedpoints, twodpoints_cam0, gray = calib_cam.detect_checker_corners([cam0_img_files], custom_pts_path = "20110110-tm1-ub-maca/Calibration/Extrinseque/cam0/manual_pts_data.pkl")
    threedpoints, twodpoints_cam1, gray = calib_cam.detect_checker_corners([cam1_img_files], custom_pts_path = "20110110-tm1-ub-maca/Calibration/Extrinseque/cam1/manual_pts_data.pkl")



    uv_0 = np.array(twodpoints_cam0)[0].reshape((36,2))
    uv_1 = np.array(twodpoints_cam1)[0].reshape((36,2))



    #load pickled calibration result
    with open('calibration_result.pickle', 'rb') as file:
        calib_result= pickle.load(file)
    P0,P1= calib_cam.get_projection_mat(matrix_0=calib_result['camera_matrix_c0'], matrix_1=calib_result['camera_matrix_c1'], R=calib_result['c0_R_c1'], T=calib_result['c0_t_c1'])


    p3ds, dist3d_list = calib_cam.triangulate_pts(projection_mat_0=P0, projection_mat_1=P1, img_pts_0=uv_0, img_pts_1=uv_1, frame_0=frame_0, frame_1=frame_1, show_triangulation_result=True, verbose=True)


    #RMSE for checkerboard (includes custom code. make changes as needed)

    #remove the distance between 2 columns because the function calculate distance between every consecutive points
    for pt_id in range(len(dist3d_list)):

        if pt_id >= len(dist3d_list): break

        if dist3d_list[pt_id] > 8: #removing any distance value that's greater than 8 
            print("popped",dist3d_list.pop(pt_id)) 

    actual_checker_size= 4
    actualy_size_arr= np.zeros((len(dist3d_list))) + actual_checker_size

    MSE = np.square(np.subtract(actualy_size_arr,dist3d_list)).mean() #in cm
    RMSE = np.sqrt(MSE) #in cm

    print("  ")
    print("MSE", MSE)
    print("RMSE", RMSE)


# test_calibration()



# #[[-0.37787529  0.35057683  0.00473403  0.00141145 -0.24608929]] 
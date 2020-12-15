import random
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
import dense_correspondence_manipulation.utils.utils as utils
import yaml
utils.add_dense_correspondence_to_python_path()

import dense_correspondence
from dense_correspondence.evaluation.evaluation import *
import dense_correspondence.correspondence_tools.correspondence_plotter as correspondence_plotter
from dense_correspondence.dataset.dense_correspondence_dataset_masked import ImageType

config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 
                               'dense_correspondence', 'evaluation', 'evaluation.yaml')
config = utils.getDictFromYamlFilename(config_filename)
default_config = utils.get_defaults_config()

# utils.set_cuda_visible_devices([0])
dce = DenseCorrespondenceEvaluation(config)
DCE = DenseCorrespondenceEvaluation

network_name = "caterpillar_3"
dcn = dce.load_network_from_config(network_name)
dataset = dcn.load_training_dataset()
# DenseCorrespondenceEvaluation.evaluate_network_qualitative(dcn, dataset=dataset, randomize=True)

dataset_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                                       'dataset', 'composite',
                                       'caterpillar_upright.yaml')
# dataset_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
#                                        'training', 'training.yaml')
dataset_config = utils.getDictFromYamlFilename(dataset_config_filename)

dataset = SpartanDataset(debug=True, config=dataset_config)
dataset.debug=False




def get_relative_pose(dataset,metadata):
    X_OCa = dataset.get_pose_from_scene_name_and_idx(metadata['scene_name'], metadata['image_a_idx'])
    X_OCb = dataset.get_pose_from_scene_name_and_idx(metadata['scene_name'], metadata['image_b_idx'])
    X_CbCa = np.matmul(np.linalg.inv(X_OCb), X_OCa)
    return X_CbCa


def get_intrinsic(dataset, metadata):
    scene_directory = dataset.get_full_path_for_scene(metadata['scene_name'])
    yaml_filename = images_dir = os.path.join(scene_directory, 'images','camera_info.yaml')
    config = yaml.load(open(yaml_filename), Loader=yaml.CLoader)
    
    fx = config['camera_matrix']['data'][0]
    cx = config['camera_matrix']['data'][2]

    fy = config['camera_matrix']['data'][4]
    cy = config['camera_matrix']['data'][5]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0,0,1]])
    
    dist_coeffs = np.array(config['distortion_coefficients']['data'][:])
    return K, dist_coeffs

def get_matches(dcn, dataset, rgb_a, rgb_b, mask_a, mask_b,num_matches):
    mask_a = np.asarray(mask_a)
    mask_b = np.asarray(mask_b)

    # compute dense descriptors
    rgb_a_tensor = dataset.rgb_image_to_tensor(rgb_a)
    rgb_b_tensor = dataset.rgb_image_to_tensor(rgb_b)

    # these are Variables holding torch.FloatTensors, first grab the data, then convert to numpy
    res_a = dcn.forward_single_image_tensor(rgb_a_tensor).data.cpu().numpy()
    res_b = dcn.forward_single_image_tensor(rgb_b_tensor).data.cpu().numpy()


    # sample points on img_a. Compute best matches on img_b
    # note that this is in (x,y) format
    # TODO: if this mask is empty, this function will not be happy
    # de-prioritizing since this is only for qualitative evaluation plots
    sampled_idx_list = random_sample_from_masked_image(mask_a, num_matches)

    # list of cv2.KeyPoint
    kp1 = []
    kp2 = []
    matches = []  # list of cv2.DMatch
    diffs = []

    # placeholder constants for opencv
    diam = 0.01
    dist = 0.01

    try:
        descriptor_image_stats = dcn.descriptor_image_stats
    except:
        print "Could not find descriptor image stats..."
        print "Only normalizing pairs of images!"
        descriptor_image_stats = None

    for i in xrange(0, num_matches):
        # convert to (u,v) format
        pixel_a = [sampled_idx_list[1][i], sampled_idx_list[0][i]]
        best_match_uv, best_match_diff, norm_diffs =\
            DenseCorrespondenceNetwork.find_best_match(pixel_a, res_a,res_b)

        # be careful, OpenCV format is  (u,v) = (right, down)
        kp1.append(cv2.KeyPoint(pixel_a[0], pixel_a[1], diam))
        kp2.append(cv2.KeyPoint(best_match_uv[0], best_match_uv[1], diam))
        diffs.append(best_match_diff)
        matches.append(cv2.DMatch(i, i, dist))
    return kp1,kp2, matches, diffs

def epipolar_constraint(R,t,kp1, kp2,K,dist_coeffs, mask=None, return_vector=False):
    t=t.reshape(-1)
    #     print t
    t_x = np.array([[0, -t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    #     print t_x
    #     print R
    E =np.dot(t_x,R)
    
    kp1_array = np.array([kp1[idx].pt for idx in range(len(kp1))]).reshape(-1, 1,2)
    kp2_array = np.array([kp2[idx].pt for idx in range(len(kp2))]).reshape(-1,1, 2)
    pts_1_norm = cv2.undistortPoints(kp1_array, cameraMatrix=K, distCoeffs=dist_coeffs)
    pts_2_norm = cv2.undistortPoints(kp2_array, cameraMatrix=K, distCoeffs=dist_coeffs)
    
#     kp1_array = np.array([kp1[idx].pt for idx in range(len(kp1))]).reshape(-1, 2)
#     kp2_array = np.array([kp2[idx].pt for idx in range(len(kp2))]).reshape(-1, 2)

    # kp1_array = np.hstack([kp1_array, np.ones(kp1_array.shape[0],1)])
    # kp2_array = np.hstack([kp2_array, np.ones(kp1_array.shape[0],1)])
    mat_e = E.reshape(-1,1)
    u1 = pts_1_norm[:,0,0]
    u2 = pts_1_norm[:,0,1]
    v1 = pts_2_norm[:,0,0]
    v2 = pts_2_norm[:,0,1]

    A = np.stack([u1*u2, u1*v2, u1, v1*u2, v1*v2, v1, u2, v2,np.ones_like(u1)]).T
    # print A.shape
    error = np.abs(np.matmul(A,mat_e))
    # print 'error:',error
    if return_vector:
    	return error.reshape(-1)
    # print error
    # error = np.matmul(kp1_array, E, kp2_array.T)
    # return error.mean()
    if mask is not None and mask.sum():
    	return (error*mask).sum()/mask.sum()
    else:
    	return error.mean()

def get_pose_essential_opencv(kp1,kp2,K,dist_coeffs):
	pp = K[:2,2]
	focal_length = (K[0][0]+K[1][1])/2

	kp1_array = np.array([kp1[idx].pt for idx in range(len(kp1))]).reshape(-1, 1,2)
	kp2_array = np.array([kp2[idx].pt for idx in range(len(kp2))]).reshape(-1,1, 2)
	pts_1_norm = cv2.undistortPoints(kp1_array, cameraMatrix=K, distCoeffs=dist_coeffs)
	pts_2_norm = cv2.undistortPoints(kp2_array, cameraMatrix=K, distCoeffs=dist_coeffs)

	pts1 = np.int32(kp1_array)
	pts2 = np.int32(kp2_array)
	F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC) #,ransacReprojThreshold=

	E = np.matmul(K.T, np.matmul(F, K))

	# E, mask = cv2.findEssentialMat(pts_1_norm, pts_2_norm,cameraMatrix=K, method=cv2.RANSAC, threshold=2/focal_length)
	_, R_est, t_est, mask2 = cv2.recoverPose(E, kp1_array, kp2_array, cameraMatrix=K, mask=mask)
	return R_est, t_est, mask

def homography_method(kp1,kp2,K,dist_coeffs):
	pp = K[:2,2]
	focal_length = (K[0][0]+K[1][1])/2

	kp1_array = np.array([kp1[idx].pt for idx in range(len(kp1))]).reshape(-1, 1,2)
	kp2_array = np.array([kp2[idx].pt for idx in range(len(kp2))]).reshape(-1,1, 2)
	pts_1_norm = cv2.undistortPoints(kp1_array, cameraMatrix=K, distCoeffs=dist_coeffs)
	pts_2_norm = cv2.undistortPoints(kp2_array, cameraMatrix=K, distCoeffs=dist_coeffs)

	H, mask = cv2.findEssentialMat(pts_1_norm, pts_2_norm,method=cv2.RANSAC)
	# print H
	_, R_est, t_est, _ = cv2.decomposeHomographyMat(H, K)
	# print R_est, t_est
	return R_est, t_est, mask

def dense_ransac(image_a,image_b,mask_a,mask_b,K,dist_coeffs=None, num_matches = 100, plot_match=True):
	pp = K[:2,2]
	focal_length = (K[0][0]+K[1][1])/2

	kp1,kp2, matches,diffs = get_matches(dcn, dataset, image_a,image_b, mask_a,mask_b,num_matches)

	
	R_est, t_est, mask = get_pose_essential_opencv(kp1,kp2,K,dist_coeffs)
	# R_est, t_est, mask = homography_method(kp1,kp2,K,dist_coeffs)
	
	print 'accept rate: ', mask.sum()/mask.shape[0]
	print 'epipolar constraints: ', epipolar_constraint(R_est, t_est, kp1,kp2,K,dist_coeffs,mask)
	# X_real = get_relative_pose(dataset, metadata)
	# # X_real = np.linalg.inv(X_real)
	# R_real = X_real[:3,:3]
	# t_real = X_real[:3,3]
	# R_real,t_real
	# print(np.matmul(R_est, np.linalg.inv(R_real)))
	# print(t_real/np.linalg.norm(t_real), t_est.reshape(-1))
	if plot_match:
		gray_a_numpy = cv2.cvtColor(np.asarray(image_a), cv2.COLOR_BGR2GRAY)
		gray_b_numpy = cv2.cvtColor(np.asarray(image_b), cv2.COLOR_BGR2GRAY)
		draw_params = dict( singlePointColor = (255,0,0),matchesMask = mask.reshape(-1).tolist(), flags=0)
		img3 = cv2.drawMatches(gray_a_numpy, kp1, gray_b_numpy, kp2, matches, outImg=gray_b_numpy, **draw_params)
		# fig, axes = plt.subplots(nrows=1, ncols=1)
		# fig.set_figheight(10)
		# fig.set_figwidth(15)
		# axes.imshow(img3)
		# plt.show()
		return R_est,t_est,img3
	return R_est,t_est

# .

def cal_essential_cov(pts_1_norm,pts_2_norm,diffs,mask, use_cov=False):
	'''
	This method is currently not working
	'''
	mask = mask!=0
	pts_1_norm = pts_1_norm[mask[:,0],:,:]
	pts_2_norm = pts_2_norm[mask[:,0],:,:]
	diffs = np.array(diffs)[mask[:,0]]
	Q = np.diag(1/diffs**2)

	if not use_cov:
		Q = np.eye(diffs.shape[0])

	u1 = pts_1_norm[:,0,0]
	u2 = pts_1_norm[:,0,1]
	v1 = pts_2_norm[:,0,0]
	v2 = pts_2_norm[:,0,1]

	A = np.stack([u1*u2, u1*v2, u1, v1*u2, v1*v2, v1, u2, v2,np.ones_like(u1)]).T

	print A.shape
	print Q.shape

	w,v = np.linalg.eigh(np.matmul(A.T,Q).dot(A))
	# print 'eigen value:',w

	E = v[:,0].reshape(3,3)

	U, s, Vt = np.linalg.svd(E)

	E = np.matmul(U,np.diag([(s[0]+s[1])/2.,(s[0]+s[1])/2.,0 ]),Vt)

	return E

def cal_essential_cov2(kp1,kp2,K,dist_coeffs,diffs,mask):
	'''
	This method is currently not working
	'''
	# only for 8-point method
	assert np.array(mask).sum()==8
	pp = K[:2,2]
	focal_length = (K[0][0]+K[1][1])/2

	kp1_array = np.array([kp1[idx].pt for idx in range(len(kp1))]).reshape(-1, 1,2)
	kp2_array = np.array([kp2[idx].pt for idx in range(len(kp2))]).reshape(-1,1, 2)
	pts_1_norm = cv2.undistortPoints(kp1_array, cameraMatrix=K, distCoeffs=dist_coeffs)
	pts_2_norm = cv2.undistortPoints(kp2_array, cameraMatrix=K, distCoeffs=dist_coeffs)

	mask = mask!=0
	pts_1_norm = pts_1_norm[mask[:,0],:,:]
	pts_2_norm = pts_2_norm[mask[:,0],:,:]
	diffs = np.array(diffs)[mask[:,0]]
	# Q = np.diag(1/diffs**2)

	pts1 = np.int32(kp1_array[mask[:,0],:,:])
	pts2 = np.int32(kp2_array[mask[:,0],:,:])
	F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC) #,ransacReprojThreshold=

	E = np.matmul(K.T, np.matmul(F, K))

	E = E*diffs.reshape(3,3)**2
	U, s, Vt = np.linalg.svd(E)
	E = np.matmul(U,np.diag([(s[0]+s[1])/2.,(s[0]+s[1])/2.,0 ]),Vt)

	# E, mask = cv2.findEssentialMat(pts_1_norm, pts_2_norm,cameraMatrix=K, method=cv2.RANSAC, threshold=2/focal_length)
	_, R_est, t_est, mask2 = cv2.recoverPose(E, kp1_array, kp2_array, cameraMatrix=K, mask=mask)
	return R_est, t_est, mask

	return E

def revised_ransac(kp1, kp2, K, dist_coeffs,diffs, num_iter = 500): #sigma_0=sigma_0, epsilon=epsilon
	kp1_array = np.array([kp1[idx].pt for idx in range(len(kp1))]).reshape(-1, 1,2)
	kp2_array = np.array([kp2[idx].pt for idx in range(len(kp2))]).reshape(-1,1, 2)
	pts_1_norm = cv2.undistortPoints(kp1_array, cameraMatrix=K, distCoeffs=dist_coeffs)
	pts_2_norm = cv2.undistortPoints(kp2_array, cameraMatrix=K, distCoeffs=dist_coeffs)
	diffs = np.array(diffs)

	max_score = -np.inf

	for i in range(num_iter):
		selected_idx = np.random.choice(list(range(kp1_array.shape[0])),8)
		kp1_array_new = kp1_array[selected_idx,:,:]
		kp2_array_new = kp2_array[selected_idx,:,:]


		# E, mask_temp = cv2.findEssentialMat(pts_1_norm_new, pts_2_norm_new,cameraMatrix=K, method=cv2.LMEDS)
		pts1 = np.int32(kp1_array_new)
		pts2 = np.int32(kp2_array_new)
		# F, mask_temp = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
		F, mask_temp = cv2.findFundamentalMat(pts1,pts2,cv2.LMEDS)
		if F is None:
			# print 'skip:',i
			continue

		E = np.matmul(K.T, np.matmul(F, K))

		# print , mask_temp.shape
		mask = np.zeros((kp1_array.shape[0],1)).astype(mask_temp.dtype)
		mask[selected_idx] = 1
		_, R_est, t_est, mask2 = cv2.recoverPose(E, kp1_array, kp2_array, cameraMatrix=K, mask=mask)
		error = epipolar_constraint(R_est, t_est, kp1,kp2,K,dist_coeffs,mask=None, return_vector = True)

		p = 1/np.sqrt(np.pi*2)/(sigma_0*diffs) * np.exp(-0.5*(error/(sigma_0*diffs))**2)
		# p  = np.exp(-0.5*(error/(sigma_0*diffs))**2)
		p=np.clip(p,epsilon,1.)
		# print p.shape
		score = np.log(p).sum()-p.shape[0]*np.log(epsilon)
		# score = p.sum() - p.shape[0]*epsilon#np.log(p).sum()-p.shape[0]*np.log(epsilon)
		# print i,score

		if score>max_score:
			best_E = E
			# best_mask = (p>epsilon).astype(mask_temp.dtype)
			# best_mask = mask#(p>epsilon).astype(mask_temp.dtype)
			best_mask = get_inliers(error).reshape((-1,1)).astype(mask_temp.dtype)

	return best_E,best_mask

def get_inliers(error,rate=0.1, threshold=0.01):
	# print error.mean()
	
	# 
	mask = error<0.01
	if mask.sum()<error.shape[0]*rate:
		cutoff_idx = error.argsort()[int(error.shape[0]*rate)]
		mask = error<=error[cutoff_idx]
	return mask




def get_pose_essential2(kp1,kp2,K,dist_coeffs, diffs):
	pp = K[:2,2]
	focal_length = (K[0][0]+K[1][1])/2

	kp1_array = np.array([kp1[idx].pt for idx in range(len(kp1))]).reshape(-1, 1,2)
	kp2_array = np.array([kp2[idx].pt for idx in range(len(kp2))]).reshape(-1,1, 2)
	pts_1_norm = cv2.undistortPoints(kp1_array, cameraMatrix=K, distCoeffs=dist_coeffs)
	pts_2_norm = cv2.undistortPoints(kp2_array, cameraMatrix=K, distCoeffs=dist_coeffs)

	# E, mask = cv2.findEssentialMat(pts_1_norm, pts_2_norm,cameraMatrix=K, method=cv2.RANSAC, threshold=2/focal_length)
	# _, R_est, t_est, mask2 = cv2.recoverPose(E, kp1_array, kp2_array, cameraMatrix=K, mask=mask)
	# print R_est, t_est

	E, mask = revised_ransac(kp1,kp2,K, dist_coeffs,diffs)
	_, R_est, t_est, mask2 = cv2.recoverPose(E, kp1_array, kp2_array, cameraMatrix=K, mask=mask)
	# print R_est, t_estf

	# E = cal_essential_cov(pts_1_norm, pts_2_norm, diffs, mask)
	# _, R_est, t_est, mask2 = cv2.recoverPose(E, kp1_array, kp2_array, cameraMatrix=K, mask=mask)
	# print R_est, t_est
	# raw_input()
	return R_est, t_est, mask
	pass

def dense_ransac2(image_a,image_b,mask_a,mask_b,K,dist_coeffs=None, num_matches = 100, plot_match=True):
	pp = K[:2,2]
	focal_length = (K[0][0]+K[1][1])/2

	kp1,kp2, matches,diffs = get_matches(dcn, dataset, image_a,image_b, mask_a,mask_b,num_matches)

	
	R_est, t_est, mask = get_pose_essential2(kp1,kp2,K,dist_coeffs, diffs)
	# R_est, t_est, mask = homography_method(kp1,kp2,K,dist_coeffs)
	
	print 'accept rate: ', mask.sum()/mask.shape[0]
	print 'epipolar constraints: ', epipolar_constraint(R_est, t_est, kp1,kp2,K,dist_coeffs,mask)
	# X_real = get_relative_pose(dataset, metadata)
	# # X_real = np.linalg.inv(X_real)
	# R_real = X_real[:3,:3]
	# t_real = X_real[:3,3]
	# R_real,t_real
	# print(np.matmul(R_est, np.linalg.inv(R_real)))
	# print(t_real/np.linalg.norm(t_real), t_est.reshape(-1))
	if plot_match:
		gray_a_numpy = cv2.cvtColor(np.asarray(image_a), cv2.COLOR_BGR2GRAY)
		gray_b_numpy = cv2.cvtColor(np.asarray(image_b), cv2.COLOR_BGR2GRAY)
		img3 = cv2.drawMatches(gray_a_numpy, kp1, gray_b_numpy, kp2, matches, flags=2, outImg=gray_b_numpy,  matchesMask = mask.reshape(-1).tolist())
		# fig, axes = plt.subplots(nrows=1, ncols=1)
		# fig.set_figheight(10)
		# fig.set_figwidth(15)
		# axes.imshow(img3)
		# plt.show()
		return R_est,t_est,img3
	return R_est,t_est

def dense_ransac_comp(image_a,image_b,mask_a,mask_b,K,dist_coeffs, metadata, num_matches = 100, plot_match=True, num_iter=10):
	pp = K[:2,2]
	focal_length = (K[0][0]+K[1][1])/2

	min_error2 = np.inf
	min_error1 = np.inf

	for i in range(num_iter):
		# print 'round: ',i

		kp1,kp2, matches,diffs = get_matches(dcn, dataset, image_a,image_b, mask_a,mask_b,num_matches)

		
		R_est, t_est, mask = get_pose_essential2(kp1,kp2,K,dist_coeffs, diffs)
		if mask.sum()==0: 
			# pass
			print '2,masked'

		# dtheta, dt = cal_error(dataset, metadata,R_est, t_est)
		error_vector = epipolar_constraint(R_est, t_est, kp1,kp2,K,dist_coeffs,mask,return_vector=True)
		diffs = np.array(diffs)
		weigtht = (1./np.array(diffs+0.01)**2).reshape(-1)
		error2 = (error_vector*weigtht*mask).sum()/(weigtht*mask).sum()
		if error2<min_error2:
			min_error2 = error2
			best_R_est2, best_t_est2 = R_est, t_est
		# print 'old ransac'

		R_est, t_est, mask = get_pose_essential_opencv(kp1,kp2,K,dist_coeffs)
		if mask.sum()==0: 
			pass
			# print '1,masked '
		# print 'accept rate: ', mask.sum()/mask.shape[0]
		# raw_input()
		error1 = epipolar_constraint(R_est, t_est, kp1,kp2,K,dist_coeffs,mask)
		if error1<min_error1:
			min_error1 = error1
			best_R_est1, best_t_est1 = R_est, t_est
		# dtheta, dt = cal_error(dataset, metadata,R_est, t_est)

	return best_R_est1, best_t_est1, best_R_est2, best_t_est2
		# print 'accept rate: ', mask.sum()/mask.shape[0]
		# print 'epipolar constraints: ', epipolar_constraint(R_est, t_est, kp1,kp2,K,dist_coeffs,mask)

def sift_ransac(image_a,image_b,mask_a,mask_b,K,dist_coeffs=None, plot_match=False, use_mask = True):
	sift = cv2.xfeatures2d.SIFT_create()
	# find the keypoints and descriptors with SIFT
	if use_mask:
		kp1, des1 = sift.detectAndCompute(np.array(image_a),np.array(mask_a))
		kp2, des2 = sift.detectAndCompute(np.array(image_b),np.array(mask_b))
	else:
		kp1, des1 = sift.detectAndCompute(np.array(image_a),None)
		kp2, des2 = sift.detectAndCompute(np.array(image_b),None)
	# FLANN parameters
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)
	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)

	# bf = cv2.BFMatcher()
	# matches = bf.knnMatch(des1,des2, k=2)

	pts1 = []
	pts2 = []
	pts1_cv = []
	pts2_cv = []
	good_matches = []
	# ratio test as per Lowe's paper
	# dist = 0.01

	matchesMask = [[0,0] for i in xrange(len(matches))]

	for i,(m,n) in enumerate(matches):
	    if m.distance < 0.8*n.distance:
	    	# print i
	        pts2.append(kp2[m.trainIdx].pt)
	        # print kp2[m.trainIdx].pt

	        pts1.append(kp1[m.queryIdx].pt)
	        # print kp1[m.queryIdx].pt
	        pts2_cv.append(kp2[m.trainIdx])
	        pts1_cv.append(kp1[m.queryIdx])
	        good_matches.append(m)
	        matchesMask[i]=[1,1]

	# kp1_array = np.array([kp1[idx].pt for idx in range(len(kp1))]).reshape(-1, 1,2)
	# kp2_array = np.array([kp2[idx].pt for idx in range(len(kp2))]).reshape(-1,1, 2)
	# pts_1_norm = cv2.undistortPoints(kp1_array, cameraMatrix=K, distCoeffs=dist_coeffs)
	# pts_2_norm = cv2.undistortPoints(kp2_array, cameraMatrix=K, distCoeffs=dist_coeffs)
	pts1 = np.int32(pts1)
	pts2 = np.int32(pts2)
	F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
	if F is None:
		F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

	try:
		E = np.matmul(K.T, np.matmul(F, K))
	except:
		# print F.shape, K.shape
		print 'Failed, use random transformation'
		E = np.random.rand(3,3)
		mask = np.ones((pts1.shape[0],1)).astype(np.uint8)

	_, R_est, t_est, _ = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K, mask=mask)

	# print 'accept rate: ', mask.sum()/mask.shape[0]
	# print 'epipolar constraints: ', epipolar_constraint(R_est, t_est, pts1_cv,pts2_cv,K,dist_coeffs,mask)
	# We select only inlier points
	# pts1 = pts1[mask.ravel()==1]
	# pts2 = pts2[mask.ravel()==1]

	# print len(pts1_cv), len(pts2_cv), len(new_matches), len(mask)
	# print pts1_cv
	# print pts2_cv
	# print new_matches
	if plot_match:
		# draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0),matchesMask = mask.reshape(-1).tolist(), flags=2)
		draw_params = dict( singlePointColor = (255,0,0),matchesMask = mask.reshape(-1).tolist(), flags=0)

		gray_a_numpy = cv2.cvtColor(np.asarray(image_a), cv2.COLOR_BGR2GRAY)
		gray_b_numpy = cv2.cvtColor(np.asarray(image_b), cv2.COLOR_BGR2GRAY)
		img3 = cv2.drawMatches(gray_a_numpy, kp1, gray_b_numpy, kp2, good_matches, outImg=gray_b_numpy, **draw_params)
		return R_est, t_est, img3
		# fig, axes = plt.subplots(nrows=1, ncols=1)
		# fig.set_figheight(10)
		# fig.set_figwidth(15)
		# axes.imshow(img3)
		# plt.show()

	return R_est, t_est
	pass

def cal_error(dataset, metadata, R_est, t_est):
	X_real = get_relative_pose(dataset, metadata)
	# X_real = np.linalg.inv(X_real)
	R_real = X_real[:3,:3]

	T0 = np.diag([1,-1,-1])
	# R_real = np.matmul(T0, R_real)
	t_real = X_real[:3,3]
	dR = np.matmul(R_est, np.linalg.inv(R_real))
	# print(dR)
	dtheta = np.arccos((np.trace(dR)-1)/2)
	# print(dtheta)
	# print(t_real/np.linalg.norm(t_real), t_est.reshape(-1))
	dt = 1-abs(np.dot(t_real/np.linalg.norm(t_real), t_est.reshape(-1)))
	# print(dt)
	# t_est = np.dot(T0, t_est)
	# dt = 1-abs(np.dot(t_real/np.linalg.norm(t_real), t_est.reshape(-1)))
	# print(dt)
	return dtheta,dt

def check_valid(mask, threshold=0.1):
	'''
	check if the object is top small
	'''
	mask = np.array(mask)
	ratio = mask.sum()/float(mask.shape[0]*mask.shape[1])
	print 'mask ratio: ', ratio
	if ratio<threshold:
		return False
	else:
		return True

def get_random_two():
	match_type, image_a_rgb, image_b_rgb, \
	matches_a, matches_b, masked_non_matches_a, \
	masked_non_matches_a, non_masked_non_matches_a, \
	non_masked_non_matches_b, blind_non_matches_a, \
	blind_non_matches_b, metadata = dataset.get_single_object_within_scene_data()	

	image_a, _, mask_a, pose_a = dataset.get_rgbd_mask_pose(metadata['scene_name'], metadata['image_a_idx'])
	image_b, _, mask_b, pose_b = dataset.get_rgbd_mask_pose(metadata['scene_name'], metadata['image_b_idx'])
	return image_a, mask_a, image_b, mask_b, metadata

def get_consecutive_two():
	metadata = dict()

	# pick a scene
	scene_name = dataset.get_random_scene_name()
	metadata['scene_name'] = scene_name

	# image_a_idx = dataset.get_random_image_index(scene_name)
	pose_data = dataset.get_pose_data(scene_name)
	image_idxs = pose_data.keys()
	image_idxs = sorted(image_idxs, key=lambda x:int(x))
	# print image_idxs
	id1 = np.random.choice(list(range(len(image_idxs)-1)))
	image_a_idx = image_idxs[id1]
	image_b_idx = image_idxs[id1+1]

	image_a, _, mask_a, pose_a = dataset.get_rgbd_mask_pose(scene_name, image_a_idx)
	metadata['image_a_idx'] = image_a_idx

	image_b, _, mask_b, pose_b = dataset.get_rgbd_mask_pose(scene_name, image_b_idx)
	metadata['image_b_idx'] = image_b_idx
	return image_a, mask_a, image_b, mask_b, metadata

def get_fix_two():
	# metadata = {'scene_name': '2018-04-10-16-08-46', 'image_a_idx': 838, 'image_b_idx': 842}
	metadata ={'image_a_idx': 268, 'object_id': 'caterpillar', 'scene_name': '2018-04-16-15-23-41', 'object_id_int': 0, 'type': 0, 'image_b_idx': 2426}
	image_a, _, mask_a, pose_a = dataset.get_rgbd_mask_pose(metadata['scene_name'], metadata['image_a_idx'])
	image_b, _, mask_b, pose_b = dataset.get_rgbd_mask_pose(metadata['scene_name'], metadata['image_b_idx'])
	return image_a, mask_a, image_b, mask_b, metadata

def get_from_metadata(metadata):
	image_a, _, mask_a, pose_a = dataset.get_rgbd_mask_pose(metadata['scene_name'], metadata['image_a_idx'])
	image_b, _, mask_b, pose_b = dataset.get_rgbd_mask_pose(metadata['scene_name'], metadata['image_b_idx'])
	return image_a, mask_a, image_b, mask_b, metadata

def eval_one(num_iters = 10):
	image_a, mask_a, image_b, mask_b, metadata = get_random_two()
	print metadata

	# check_valid(mask_a)
	# check_valid(mask_b)



	K, dist_coeffs = get_intrinsic(dataset, metadata)

	# for i in range(num_iters):
	# 	R_est,t_est = dense_ransac2(image_a,image_b,mask_a,mask_b,K,dist_coeffs)
	# 	cal_error(dataset, metadata,R_est, t_est)
	R_est0,t_est0 = sift_ransac(image_a,image_b,mask_a,mask_b,K,dist_coeffs)
	dtheta0, dt0 = cal_error(dataset, metadata,R_est0, t_est0)

	R_est1, t_est1, R_est2, t_est2 = dense_ransac_comp(image_a,image_b,mask_a,mask_b,K,dist_coeffs,metadata)
	dtheta1, dt1 = cal_error(dataset, metadata,R_est1, t_est1)
	dtheta2, dt2 = cal_error(dataset, metadata,R_est2, t_est2)

	print '0: ', dtheta0, dt0
	print '1: ', dtheta1, dt1
	print '2: ', dtheta2, dt2

	return (dtheta0, dt0), (dtheta1, dt1), (dtheta2, dt2)

def eval_test_cases(filename):
	metadata_list = pkl.load(open(filename,'rb'))

	result = []

	for i,metadata in enumerate(metadata_list):
		image_a, mask_a, image_b, mask_b, metadata = get_from_metadata(metadata)
		print i,metadata

		K, dist_coeffs = get_intrinsic(dataset, metadata)

		# for i in range(num_iters):
		# 	R_est,t_est = dense_ransac2(image_a,image_b,mask_a,mask_b,K,dist_coeffs)
		# 	cal_error(dataset, metadata,R_est, t_est)
		R_est0,t_est0 = sift_ransac(image_a,image_b,mask_a,mask_b,K,dist_coeffs)
		dtheta0, dt0 = cal_error(dataset, metadata,R_est0, t_est0)

		R_est1, t_est1, R_est2, t_est2 = dense_ransac_comp(image_a,image_b,mask_a,mask_b,K,dist_coeffs,metadata)
		dtheta1, dt1 = cal_error(dataset, metadata,R_est1, t_est1)
		dtheta2, dt2 = cal_error(dataset, metadata,R_est2, t_est2)

		print '0: ', dtheta0, dt0
		print '1: ', dtheta1, dt1
		print '2: ', dtheta2, dt2

		result.append((dtheta0, dt0,dtheta1, dt1,dtheta2, dt2))

	dtheta0 = np.mean([x[0] for x in result])
	dt0 = np.mean([x[1] for x in result])
	dtheta1 = np.mean([x[2] for x in result])
	dt1 = np.mean([x[3] for x in result])
	dtheta2 = np.mean([x[4] for x in result])
	dt2 = np.mean([x[5] for x in result])

	print 'final result:'
	print 'sigma_0:',sigma_0
	print 'epsilon',epsilon
	print '0: ', dtheta0, dt0
	print '1: ', dtheta1, dt1
	print '2: ', dtheta2, dt2


def gen_test_set():
	metadata_list = []
	for i in range(100):
		image_a, mask_a, image_b, mask_b, metadata = get_consecutive_two()
		print metadata
		metadata_list.append(metadata)
	pkl.dump(metadata_list, open('consecutive_two_100_test_cases.pkl','wb'))

import pickle as pkl
import time
if __name__ == '__main__':
	sigma_0 = 1.
	epsilon = 0.1
	# eval_test_cases('consecutive_two_100_test_cases.pkl')
	eval_test_cases('random_two_100_test_cases.pkl')
	# sigma_0_list = [1,3]
	# epsilon_list = [0.01,0.03,0.1,0.2]

	# # sigma_0_epsilon_list = [(0.3,0.01)]

	# for sigma_0 in sigma_0_list:
	# 	for epsilon in epsilon_list:
	# 		eval_test_cases('random_two_100_test_cases.pkl')
	# gen_test_set()
	# t1 = time.time()
	# np.random.seed(5)
	# torch.random.manual_seed(3)
	# eval_one()
	# t2 = time.time()

	# print 'time: ', t2-t1





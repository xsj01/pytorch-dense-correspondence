from pose_estimation import *

def eval():
	# image_a, mask_a, image_b, mask_b, metadata = get_random_two()

	filename = 'random_two_100_test_cases.pkl'
	# filename = 'consecutive_two_100_test_cases.pkl'
	metadata_list = pkl.load(open(filename,'rb'))
	# print metadata
	metadata = metadata_list[44]

	# check_valid(mask_a)
	# check_valid(mask_b)
	image_a, mask_a, image_b, mask_b, metadata = get_from_metadata(metadata)
	print metadata

	K, dist_coeffs = get_intrinsic(dataset, metadata)

	# for i in range(num_iters):
	# 	R_est,t_est = dense_ransac2(image_a,image_b,mask_a,mask_b,K,dist_coeffs)
	# 	cal_error(dataset, metadata,R_est, t_est)
	R_est0,t_est0,img0 = sift_ransac(image_a,image_b,mask_a,mask_b,K,dist_coeffs,plot_match=True)
	dtheta0, dt0 = cal_error(dataset, metadata,R_est0, t_est0)

	R_est1, t_est1,img1 = dense_ransac(image_a,image_b,mask_a,mask_b,K,dist_coeffs,plot_match=True)
	dtheta1, dt1 = cal_error(dataset, metadata,R_est1, t_est1)
	# dtheta2, dt2 = cal_error(dataset, metadata,R_est2, t_est2)

	print '0: ', dtheta0, dt0
	print '1: ', dtheta1, dt1
	# print '2: ', dtheta2, dt2
	fig, axes = plt.subplots(nrows=1, ncols=2)
	fig.set_figheight(4)
	fig.set_figwidth(20)
	axes[0].imshow(img0)
	axes[0].axis('off')
	axes[1].imshow(img1)
	axes[1].axis('off')


	plt.show()

if __name__ == '__main__':
	eval()

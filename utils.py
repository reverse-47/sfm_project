import numpy as np 
import cv2 

def SerializeKeypoints(kp): 
   """Serialize list of keypoint objects for pickle saving"""
   return [(kp_.pt, kp_.size, kp_.angle, kp_.response, kp_.octave, kp_.class_id) 
           for kp_ in kp]

def DeserializeKeypoints(kp): 
    """Deserialize list of keypoint objects to opencv format"""
    return [cv2.KeyPoint(x=point[0][0], y=point[0][1], size=point[1], 
                        angle=point[2], response=point[3], octave=point[4], 
                        class_id=point[5]) for point in kp]

def SerializeMatches(matches): 
   """Serialize matches for pickle saving"""
   return [(match.queryIdx, match.trainIdx, match.imgIdx, match.distance) 
           for match in matches]

def DeserializeMatches(matches): 
   """Deserialize matches to opencv format"""
   return [cv2.DMatch(match[0], match[1], match[2], match[3]) 
           for match in matches]

def GetAlignedMatches(kp1, desc1, kp2, desc2, matches):
   """Align keypoints so rows correspond between images"""
   matches = sorted(matches, key=lambda x: x.distance)
   
   img1idx = np.array([m.queryIdx for m in matches])
   img2idx = np.array([m.trainIdx for m in matches])

   kp1_ = (np.array(kp1))[img1idx]
   kp2_ = (np.array(kp2))[img2idx]

   img1pts = np.array([kp.pt for kp in kp1_])
   img2pts = np.array([kp.pt for kp in kp2_])

   return img1pts, img2pts

def pts2ply(pts, colors, filename='out.ply'): 
   """Save ndarray of 3D coordinates in meshlab format"""
   with open(filename, 'w') as f:
       f.write('ply\n')
       f.write('format ascii 1.0\n')
       f.write(f'element vertex {pts.shape[0]}\n')
       
       f.write('property float x\n')
       f.write('property float y\n')
       f.write('property float z\n')
       
       f.write('property uchar red\n')
       f.write('property uchar green\n')
       f.write('property uchar blue\n')
       
       f.write('end_header\n')
       
       colors = colors.astype(int)
       for pt, cl in zip(pts, colors):
           f.write(f'{pt[0]} {pt[1]} {pt[2]} {cl[0]} {cl[1]} {cl[2]}\n')

def DrawCorrespondences(img, ptsTrue, ptsReproj, ax, drawOnly=50): 
   """Draw correspondence between ground truth and reprojected points"""
   ax.imshow(img)
   
   randidx = np.random.choice(ptsTrue.shape[0], size=(drawOnly,), replace=False)
   ptsTrue_, ptsReproj_ = ptsTrue[randidx], ptsReproj[randidx]
   
   colors = np.random.rand(drawOnly, 3)
   
   ax.scatter(ptsTrue_[:,0], ptsTrue_[:,1], marker='x', c='r', 
             linewidths=.1, label='Ground Truths')
   ax.scatter(ptsReproj_[:,0], ptsReproj_[:,1], marker='x', c='b',
             linewidths=.1, label='Reprojected')
   ax.legend()

   return ax
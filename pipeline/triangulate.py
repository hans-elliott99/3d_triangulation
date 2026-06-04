

import numpy as np



def compute_epipoles(F: np.array):
    '''
    Compute epipoles from a 3x3 fundamental matrix F
    The epipoles are the null spaces of F and F^T respectively
    '''
    U, S, Vt = np.linalg.svd(F)
    e1 = Vt[-1]   # right null space of F (last col of V/last row of Vt)
    e2 = U[:, -1] # left null space of F (last col of U)
    return e1, e2


class ImageData:
    def __init__(self, data):
        self.data = data
        if data.shape[1] != 4:
            raise ValueError(
                f'Expected data with 4 columns (x1, y1, x2, y2), got {data.shape[1]} columns'
            )
        if data.shape[0] < 8:
            raise ValueError(
                f'At least 8 matches are required to compute the fundamental matrix, got {data.shape[0]} matches'
            )
        self.F = self._compute_fundamental_matrix(data)
        self.P1, self.P2 = self._compute_camera_matrices(self.F)
    
    def _compute_fundamental_matrix(self, data):
        '''
        Normalized 8-point algorithm
        Input: n >= 8 point correspondences x = (x1, y1) <--> x' = (x2, y2)
        Output: A fundamental matrix F s.t. (x')^T F x = 0 for each correspondence
        '''

        # Normalization transformations:
        #   Translate and scale the points so that the centroid is at the origin
        #   and the RMS distance to the origin is sqrt(2).
        centers = np.mean(data, axis=0)

        rms1 = np.sqrt(np.mean((data[:, 0:2] - centers[0:2]) ** 2))
        rms2 = np.sqrt(np.mean((data[:, 2:4] - centers[2:4]) ** 2))
        if rms1 == 0 or rms2 == 0:
            raise ValueError('All points are the same in one of the images, cannot compute fundamental matrix')
        scales = np.array([np.sqrt(2) / rms1, np.sqrt(2) / rms2])

        T = np.array([
            # s, 0, -s * cx
            [scales[0], 0, -scales[0] * centers[0]],
            # 0, s, -s * cy
            [0, scales[0], -scales[0] * centers[1]],
            [0, 0, 1]
        ])
        T_prime = np.array([
            [scales[1], 0, -scales[1] * centers[2]],
            [0, scales[1], -scales[1] * centers[3]],
            [0, 0, 1]
        ])

        # Construct matrix A for the 8-point algorithm
        data_norm = (data - centers) * np.array([scales[0], scales[0], scales[1], scales[1]])

        A = np.empty(shape=(data_norm.shape[0], 9))
        for i, row in enumerate(data_norm):
            x1, y1, x2, y2 = row
            A_row = np.array([
                x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1
            ])
            A[i, :] = A_row
        
        # Compute SVD of A and extract the fundamental matrix F
        U, S, Vt = np.linalg.svd(A)
        F = Vt[-1].reshape(3, 3) # last col of V ie last row of Vt

        # Enforce rank 2 constraint:
        #   Replace F with F' that minimizes ||F' - F|| subject to det(F') = 0.
        #   Equivalently:
        Uf, Sf, Vtf = np.linalg.svd(F)
        Sf[2] = 0
        F_rank2 = Uf @ np.diag(Sf) @ Vtf.T

        # Denormalize the fundamental matrix
        ## Undo the normalization transformations
        F_rank2 = T_prime.T @ F_rank2 @ T

        return F_rank2

    def _compute_camera_matrices(self, F):
        '''
        Input: A fundamental matrix F 
        Output: A pair of canonical camera matrices P1, P2, accurate up to
                projective ambiguity.
        '''
        # Compute epipoles from the fundamental matrix
        e1, e2 = compute_epipoles(F)
        # Let P = [I | 0] (3x4)
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        # Then P' = [[e2]_x F | e2],
        #   where [e2]_x is the skew-symmetric matrix of e2 s.t.
        #   [e2]_x v = e2 x v (cross prod) for any vector v.
        e2_x = np.array([
            [0, -e2[2], e2[1]],
            [e2[2], 0, -e2[0]],
            [-e2[1], e2[0], 0]
        ])
        P2 = np.hstack((e2_x @ F, e2.reshape(3, 1)))
        return P1, P2


class TriangulationOptimization:
    def __init__(self, f1, f2, a, b, c, d):
        self.f1 = f1
        self. f2 = f2
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self.coefs = self._polyn_coefficients()
        self.roots = np.roots(self.coefs)
        self.t_min = self._compute_t_min(self.roots)
    
    def evaluate_polyn(self, t):
        # for testing
        return t * (
            (self.a*t + self.b)**2 + self.f2**2 * (self.c*t + self.d)**2
        )**2 - (
            (self.a*self.d - self.b*self.c) *
            (1 + self.f1**2 * t**2)**2 *
            (self.a*t + self.b) *
            (self.c*t + self.d)
        )
    
    def _polyn_coefficients(self):
        '''
        Return the coefficients of the 6th degree polynomial g(t)
        (see scripts/expand_triangulation_polyn.py)
        '''
        f, f_prime, a, b, c, d = self.f1, self.f2, self.a, self.b, self.c, self.d
        coefs = np.zeros(shape=(7,)) #t^6, t^5, ..., t^0
        coefs[0] = -a**2*c*d*f**4 + a*b*c**2*f**4
        coefs[1] = a**4 + 2*a**2*c**2*f_prime**2 - a**2*d**2*f**4 + b**2*c**2*f**4 + c**4*f_prime**4
        coefs[2] = 4*a**3*b - 2*a**2*c*d*f**2 + 4*a**2*c*d*f_prime**2 + 2*a*b*c**2*f**2 + 4*a*b*c**2*f_prime**2 - a*b*d**2*f**4 + b**2*c*d*f**4 + 4*c**3*d*f_prime**4
        coefs[3] = 6*a**2*b**2 - 2*a**2*d**2*f**2 + 2*a**2*d**2*f_prime**2 + 8*a*b*c*d*f_prime**2 + 2*b**2*c**2*f**2 + 2*b**2*c**2*f_prime**2 + 6*c**2*d**2*f_prime**4
        coefs[4] = -a**2*c*d + 4*a*b**3 + a*b*c**2 - 2*a*b*d**2*f**2 + 4*a*b*d**2*f_prime**2 + 2*b**2*c*d*f**2 + 4*b**2*c*d*f_prime**2 + 4*c*d**3*f_prime**4
        coefs[5] = -a**2*d**2 + b**4 + b**2*c**2 + 2*b**2*d**2*f_prime**2 + d**4*f_prime**4
        coefs[6] = -a*b*d**2 + b**2*c*d
        return coefs
    
    def _evaluate_cost(self, t):
        '''
        Evaluate the cost function s(t) at a given t
        '''
        return (
            t**2 /
            (1 + t**2 * self.f1**2)
        ) + (
            (self.c * t + self.d)**2 /
            (self.a * t + self.b)**2 + self.f2**2 * (self.c * t + self.d)**2
        )
        
    def _compute_t_min(self, roots):
        t_values = [root.real for root in roots] ## shortcut: just eval at all real parts
        s_values = [self._evaluate_cost(t) for t in t_values]
        t_min = t_values[np.argmin(s_values)]
        # also eval at t --> infinity
        s_inf = 1/self.f1**2 + self.c**2 / (self.a**2 + self.f2**2 + self.f2**2 * self.c**2)
        if s_inf < t_min:
            # cost is minimized as t --> infinity, i.e., the world point X lies
            # on the baseline between the camera centers, so degenerate
            return np.inf
        return t_min

        

def closest_point_to_origin(line):
    '''
    Input: A line in homogeneous coords (a, b, c) representing ax + by + cz = 0
    Output: The point on the line closest to the origin (0,0,1) of an image plane,
        in homogeneous coordinates
    '''
    lam, mu, nu = line
    return np.array([-lam*nu, -mu*nu, lam**2 + mu**2])


class TriangulationPipeline:
    '''
    3d 
    '''
    def __init__(self, image_data: ImageData):
        self.image_data = image_data
    

    def _corrected_correspondences(self, x1, y1, x2, y2, F):
        '''
        Input: Measured point correspondences (x1, y1) <--> (x2, y2)
        Output: Corrected correspondences (x1_hat, y1_hat) <--> (x2_hat, y2_hat)
                that minimize the sum of squared distances subject to the
                epipolar constraint.
        '''

        ### x1, y1, x2, y2 = image_data.data[1, ]
        ### F = image_data.F

        # 1. Define transformation matrices to map homog coords
        #    (x1, x2, z1), (y1, y2, z2) to the origin
        T1 = np.array([
            [1, 0, -x1],
            [0, 1, -y1],
            [0, 0, 1]
        ])
        T2 = np.array([
            [1, 0, -x2],
            [0, 1, -y2],
            [0, 0, 1]
        ])
        T1_inv = np.linalg.inv(T1)
        T2_inv = np.linalg.inv(T2)

        # 2. Update F to translate its coords
        F_trl = T2_inv.T @ F @ T1_inv

        # 3. Compute right and left epipoles and normalize
        e1, e2 = compute_epipoles(F_trl)
        ## normalize s.t e1[0]^2 + e1[1]^2 = 1, similarly for e2
        e1 /= np.linalg.norm(e1[:2])
        e2 /= np.linalg.norm(e2[:2])

        # 4. Form rotation matrices
        R1 = np.array([
            [e1[0], e1[1], 0],
            [-e1[1], e1[0], 0],
            [0, 0, 1]
        ])
        R2 = np.array([
            [e2[0], e2[1], 0],
            [-e2[1], e2[0], 0],
            [0, 0, 1]
        ])

        # 5. Update F so that epipoles are at (1, 0, e1[2]), (1, 0, e2[2]),
        #    giving F special form
        F_trl = R2 @ F_trl @ R1.T

        # 6-8. Form g(t) and solve for roots of g, and then find which root
        # minimizes cost function s(t) to get t_min
        opt = TriangulationOptimization(
            f1=e1[2], f2=e2[2],
            a=F_trl[1,1], b=F_trl[1,2], c=F_trl[2,1], d=F_trl[2,2]
        )
        t_min = opt.t_min
        if np.isinf(t_min):
            raise ValueError('Degenerate configuration: t_min is infinite, world point lies on camera baseline')

        # 9. Evaluate l1(t) and l2(t), the optimal epipolar lines,
        #    and find closest points on these lines to the origin.
        #    These are the corrected points.
        ## (use opt data for convenience)
        l1 = [t_min * opt.f1, 1, -t_min]
        l2 = [-opt.f2 * (opt.c * t_min + opt.d),
              opt.a * t_min + opt.b,
              opt.c * t_min + opt.d]
        
        hat_x = closest_point_to_origin(l1)
        hat_xp = closest_point_to_origin(l2)

        # 10. Transform back to original coordinates
        hat_x = T1_inv @ R1.T @ hat_x
        hat_xp = T2_inv @ R2.T @ hat_xp

        # standardize homogeneous coords
        hat_x /= hat_x[2]
        hat_xp /= hat_xp[2]

        return hat_x, hat_xp
    
    def _linear_triangulation(self, x1, y1, x2, y2, P1, P2):
        '''
        Input: (Corrected) image points x=(x1,y1,z1) <--> xp=(x2,y2,z2) and
               camera matrices P1, P2 s.t. x = P1 X, xp = P2 X for some unkown
               world point X 
        Output: World point X
        '''

        # Setup equation to solve AX = 0 for X
        A = np.array([
            x1 * P1[2, ] - P1[0, ],
            y1 * P1[2, ] - P1[1, ],
            x2 * P2[2, ] - P2[0, ],
            y2 * P2[2, ] - P2[1, ],
        ])

        # Solve for X
        U, S, Vt = np.linalg.svd(A)
        X = Vt[-1] # last col of V ie last row of Vt

        # normalize homogeneous coordinates
        X /= X[3]
        return X

    def triangulate(self):
        n = self.image_data.data.shape[0]
        world_points = np.zeros(shape=(n, 4))
        for i, row in enumerate(self.image_data.data):
            # compute corrected correspondences
            hat_x, hat_xp = self._corrected_correspondences(
                x1=row[0], y1=row[1],
                x2=row[2], y2=row[3],
                F=self.image_data.F
            )
            # compute world point
            X = self._linear_triangulation(
                x1=hat_x[0], y1=hat_x[1],
                x2=hat_xp[0], y2=hat_xp[1],
                P1=self.image_data.P1, P2=self.image_data.P2
            )
            world_points[i, :] = X

        return world_points

        
#x1,y1,x2,y2
file = 'table'
data = np.loadtxt(f'data/{file}_matches.csv', delimiter=',', dtype=np.float32)

image_data = ImageData(data)
pipeline = TriangulationPipeline(image_data)
world = pipeline.triangulate()

## Does estimated F satisfy epipolar constraint for original correspondences?
x = np.hstack((data[:, :2], np.ones((data.shape[0], 1))))
xp = np.hstack((data[:, 2:4], np.ones((data.shape[0], 1))))
for i in range(data.shape[0]):
    x_i = x[i, ]
    xp_i = xp[i, ]
    constraint = xp_i.T @ image_data.F @ x_i
    print(f'Constraint for point {i+1}: {constraint:.6f} (should be small)')

## Does estimated F satisfy epipolar constraint for corrected correspondences?
## Note: the corrected correspondences should satisfy the epipolar constraint
##  ~exactly, since they are optimized to do so.
for i in range(data.shape[0]):
    x_i = x[i, ]
    xp_i = xp[i, ]
    hat_x, hat_xp = pipeline._corrected_correspondences(
        x1=x_i[0], y1=x_i[1],
        x2=xp_i[0], y2=xp_i[1],
        F=image_data.F
    )
    constraint = np.array([hat_xp[0], hat_xp[1], 1]).T @ image_data.F @ np.array([hat_x[0], hat_x[1], 1])
    print(f'Constraint for corrected point {i+1}: {constraint:.6f} (should be approx 0)')    

## Test reprojection (should be close to original points)
x_reproj = (image_data.P1 @ world.T).T
xp_reproj = (image_data.P2 @ world.T).T
x_reproj /= x_reproj[:, 2:3]
xp_reproj /= xp_reproj[:, 2:3]

print(image_data.data[:, :2]) # original points in image 1
print(x_reproj[:, :2])
print(image_data.data[:, 2:4]) # original points in image 2
print(xp_reproj[:, :2])

euclid = (world[:, :3].T / world[:, 3]).T
np.savetxt(f"./data/{file}_reconstruction.csv",
           euclid, delimiter=',', header='X,Y,Z', comments='')
np.savez(f"./data/{file}_cameras.npz", P1=image_data.P1, P2=image_data.P2)
import unittest
import numpy as np
from py_kernel_pca import KernelPCA, delete_arr

class TestKernelPCA(unittest.TestCase):

	def setUp(self):
		self.M = 1000
		self.N = 100
		self.test_pca = KernelPCA(-1)
		self.max_sdot = np.float32(0.0001)
		self.max_ddot = np.float64(0.0000001)
		self.K = 2
		self.test_pca2 = KernelPCA(self.K)
		self.Xd = np.random.rand(self.M, self.N)
		self.Xf = np.random.rand(self.M, self.N).astype(np.float32)
		self.Td_all = np.zeros((self.M, self.N), dtype=np.float64)	
		self.Tf_2 = np.zeros((self.M, self.K), dtype=np.float32)	
		self.Tf_all = np.zeros((self.M, self.N), dtype=np.float32)	
		self.Td_2 = np.zeros((self.M, self.K), dtype=np.float64)

	def test_type_and_shape_double_all(self):
		# test that the shape is what we think it should be
		
		Td_all = self.test_pca.fit_transform(self.Xd, verbose=True)

		self.assertIsNotNone(Td_all)

		self.assertEqual(type(Td_all[0,0]), np.float64)

		self.assertEqual(Td_all.shape, (self.M, self.N))

		self.Td_all = Td_all

	def test_ortho_double_all(self):
		# test the orthogonality of the eigenvectors
				
		for i in range(self.N-1):
			self.assertTrue(np.dot(self.Td_all[:,i], self.Td_all[:,i+1]) < self.max_ddot)

	def test_type_and_shape_all(self):
		# test that the shape is what we think it should be
		
		Tf_all = self.test_pca.fit_transform(self.Xf, verbose=True)

		self.assertIsNotNone(Tf_all)

		self.assertEqual(type(Tf_all[0,0]), np.float32)

		self.assertEqual(Tf_all.shape, (self.M, self.N))

		self.Tf_all = Tf_all

	def test_ortho_all(self):
		# test the orthogonality of the eigenvectors
				
		for i in range(self.N-1):
			self.assertTrue(np.dot(self.Tf_all[:,i], self.Tf_all[:,i+1]) < self.max_sdot)

	def test_type_and_shape_double(self):
		# test that the shape is what we think it should be
		
		Td_2 = self.test_pca2.fit_transform(self.Xd, verbose=True)

		self.assertIsNotNone(Td_2)

		self.assertEqual(type(Td_2[0,0]), np.float64)

		self.assertEqual(Td_2.shape, (self.M, self.K))

		self.Td_2 = Td_2

	def test_ortho_double(self):
		# test the orthogonality of the eigenvectors
				
		self.assertTrue(np.dot(self.Td_all[:,0], self.Td_all[:,1]) < self.max_ddot)

	def test_type_and_shape(self):
		# test that the shape is what we think it should be
		
		Tf_2 = self.test_pca2.fit_transform(self.Xf, verbose=True)

		self.assertIsNotNone(Tf_2)

		self.assertEqual(type(Tf_2[0,0]), np.float32)

		self.assertEqual(Tf_2.shape, (self.M, self.K))

		self.Tf_2 = Tf_2

	def test_ortho(self):
		# test the orthogonality of the eigenvectors
				
		self.assertTrue(np.dot(self.Tf_2[:,0], self.Tf_2[:,1]) < self.max_sdot)


			

	def test_c_contiguous_check(self):
		
		try:
			X_trash = np.random.rand(self.M, self.M)
			T_trash = self.test_pca2(X_trash.T)
			fail(msg="C-contiguous array check failed") # should not reach this line. The prev line should fail and go to the except block
		except:
			
			print '' # need some sort of code her, or else there is an error
			

	def arr_2d_check(self):

		try:
			X_trash = np.random.rand(self.M, self.Mi, 3)
			T_trash = self.test_pca2(X_trash.T)
			fail(msg="Array dimensions check failed") # should not reach this line. The prev line should fail and go to the except block
		except:
			
			print '' # need some sort of code her, or else there is an error
			
	
	def test_k_bigger_than_array_dims_and_getset(self):

		self.test_pca.set_n_components(self.N+1)

		self.assertEqual(self.test_pca.get_n_components(), self.N+1)

		X = np.random.rand(self.M, self.N).astype(np.float32)
		T = self.test_pca.fit_transform(X, verbose=True)

		self.assertEqual(self.test_pca.get_n_components(), self.N) # should have been reset internally once the algorithm saw K was bigger than N
		
		X2 = np.random.rand(self.N-2, self.N-1).astype(np.float32)
		T = self.test_pca.fit_transform(X2, verbose=True)

		self.assertEqual(self.test_pca.get_n_components(), self.N-2) # should have been reset internally once the algorithm saw K was bigger than N
	





import unittest
import numpy as np
from postpro_core import process,DEFAULTS,hue_rotate,blur

class PostproTests(unittest.TestCase):
    def setUp(self):
        self.x=np.random.default_rng(4).uniform(.1,.8,(24,32,3)).astype(np.float32)

    def test_neutral_exact(self):
        np.testing.assert_array_equal(process(self.x),self.x)
        np.testing.assert_array_equal(process(self.x,pivot=.2,sharpen_radius=10),self.x)

    def test_brightness_gamma_contrast(self):
        self.assertGreater(process(self.x,brightness=.1).mean(),self.x.mean())
        self.assertGreater(process(self.x,gamma=2).mean(),self.x.mean())
        self.assertLess(process(self.x,gamma=.5).mean(),self.x.mean())
        np.testing.assert_allclose(process(self.x,contrast=0),.5,atol=1e-6)

    def test_exposure_linear(self):
        x=np.full((2,3,3),.3)
        y=process(x,exposure=1)
        linear=((x+.055)/1.055)**2.4
        expected=1.055*(linear*2)**(1/2.4)-.055
        np.testing.assert_allclose(y,expected,atol=1e-6)

    def test_hue_and_gray(self):
        red=np.array([[[1.,0.,0.]]])
        np.testing.assert_allclose(hue_rotate(red,120),[[[0,1,0]]],atol=1e-6)
        np.testing.assert_allclose(hue_rotate(red,-120),[[[0,0,1]]],atol=1e-6)
        np.testing.assert_allclose(hue_rotate(self.x,360),self.x,atol=1e-6)
        gray=process(self.x,saturation=0)
        np.testing.assert_allclose(gray[...,0],gray[...,1],atol=1e-6)
        np.testing.assert_allclose(gray[...,1],gray[...,2],atol=1e-6)

    def test_sharpen_flat_and_edge(self):
        flat=np.full((24,32,3),.4,dtype=np.float32)
        np.testing.assert_allclose(process(flat,sharpness=2),flat,atol=1e-6)
        edge=flat.copy(); edge[:,16:]=.6
        y=process(edge,sharpness=1)
        self.assertLess(y[12,15,0],.4)
        self.assertGreater(y[12,16,0],.6)
        np.testing.assert_allclose(blur(np.ones((1,1,1)),20,lambda:None),1)

    def test_all_controls_finite(self):
        for key,val in dict(exposure=2,brightness=-.1,gamma=.4,contrast=2,shadows=.4,highlights=-.5,black_point=.1,white_point=.9,temperature=.8,tint=-.8,red_gain=2,green_gain=.3,blue_gain=2,saturation=2,vibrance=1,hue_shift=150,sharpness=3).items():
            with self.subTest(control=key):
                y=process(self.x,**{key:val})
                self.assertEqual(y.shape,self.x.shape)
                self.assertTrue(np.isfinite(y).all())
                self.assertTrue(((y>=0)&(y<=1)).all())

    def test_invalid(self):
        for kw in [dict(gamma=0),dict(white_point=.2,black_point=.3),dict(vibrance=np.nan),dict(unknown=2)]:
            with self.assertRaises(ValueError): process(self.x,**kw)
        with self.assertRaises(ValueError): process(np.full((1,1,3),np.nan))

if __name__=='__main__': unittest.main()

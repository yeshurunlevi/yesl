import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage as nd
def find_green(img):
  parameters = cv2.aruco.DetectorParameters_create()
  aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
  corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
  int_corners = np.int0(corners)
  cv2.polylines(img, int_corners, True, (0, 255, 0), 30)
  aruco_area = cv2.contourArea (corners[0])
  pixel_cm_ratio = 5*5 / aruco_area
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  mask_green = cv2.inRange(hsv, (40,150,0), (70,250,255)) 
  closed_mask_green = nd.binary_closing(mask_green, np.ones((2,2)))
  leaf_count = np.sum(np.array(closed_mask_green) >0)
  bg_count = np.sum(np.array(closed_mask_green) ==0)
  plt.imshow(closed_mask_green,cmap='gray')
  print('Leaf px count:', leaf_count, 'px')
  print('Area:', leaf_count*pixel_cm_ratio, 'cm\N{SUPERSCRIPT TWO}', 'thats',leaf_count*pixel_cm_ratio)
  # vars
DEMO_IMAGE = 'leafs.jpeg' # a demo image for the segmentation page, if none is uploaded

# main page
st.set_page_config(page_title='green segment', page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')
st.title('Image Segmentation using K-Means, by Yedidya Harris')

# side bar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] . div:first-child{
        width: 350px
    }
    
    [data-testid="stSidebar"][aria-expanded="false"] . div:first-child{
        width: 350px
        margin-left: -350px
    }    
    </style>
    
    """,
    unsafe_allow_html=True,


)

st.sidebar.title('Segmentation Sidebar')
st.sidebar.subheader('Site Pages')
# using st.cache so streamlit runs the following function only once, and stores in chache (until changed)
@st.cache()# לשמור בזכרון

# take an image, and return a resized that fits our page
def image_resize(image, width=None, height=None, inter = cv2.INTER_AREA):# change the image size to the table
    dim = None
    (h,w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)
    
    else:
        r = width/float(w)
        dim = (width, int(h*r))
        
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    
    return resized
    # add dropdown to select pages on left
app_mode = st.sidebar.selectbox('Navigate',
                                  ['About App', 'Segment an Image'])
                                  # About page
if app_mode == 'About App':
    st.markdown('In this app we will segment images using K-Means')
    
    
    # side bar
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] . div:first-child{
            width: 350px
        }

        [data-testid="stSidebar"][aria-expanded="false"] . div:first-child{
            width: 350px
            margin-left: -350px
        }    
        </style>

        """,
        unsafe_allow_html=True,


    )

    # add a video to the page
    st.video('https://www.youtube.com/watch?v=6CqRnx6Ic48')


    st.markdown('''
                ## About the app \n
                Hey, this web app is a great one to segment images using K-Means. \n
                There are many way. \n
                Enjoy! Yedidya


                ''') 
                # Run image
if app_mode == 'Segment an Image':
    
    st.sidebar.markdown('---') # adds a devider (a line)
    
    # side bar
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] . div:first-child{
            width: 350px
        }

        [data-testid="stSidebar"][aria-expanded="false"] . div:first-child{
            width: 350px
            margin-left: -350px
        }    
        </style>

        """,
        unsafe_allow_html=True,


    )
    
    # read an image from the user
    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])# bottum to up image

    # assign the uplodaed image from the buffer, by reading it in
    if img_file_buffer is not None:
        image = io.imread(img_file_buffer)
    else: # if no image was uploaded, then segment the demo image
        demo_image = DEMO_IMAGE
        image = io.imread(demo_image)

    # display on the sidebar the uploaded image
    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    
    # call the function to segment the image
    segmented_image = find_green(image)
    
    # Display the result on the right (main frame)
    st.subheader('Output Image')
    st.image(segmented_image, use_column_width=True)
    with open('requirements.txt', 'w') as f:
    f.write('''streamlit
scikit-image
opencv-contrib-python-headless
numpy
scipy
matplotlib.pyplot''')

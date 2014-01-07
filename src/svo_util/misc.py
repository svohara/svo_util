'''
Created on May 9, 2013
@author: Steve O'Hara
Miscellaneous utility functions
'''
try:
    import smtplib
    import email
except:
    print "Error importing libraries required for e-mail notification scripts."
    print "Functions which send e-mail notifications will not work."

try:    
    import pylab as pl
    from mpl_toolkits.mplot3d import Axes3D
except:
    print "Error importing pylab or matplotlib mplot3d code."
    print "3D Scatter plot function will not work."
    
import scipy as sp
import scipy.linalg as LA
import sys
import urllib

def parse_number(s, fail=None):
    '''
    Tries to convert input object s into a float. If it
    can, the float representation is returned. If not,
    then either it returns None, or returns a user-specified
    value.
    @param s: Input object to be converted to float()
    @param fail: Returned when s can't be converted to float.
    @return: Either float(s) or the value specified by fail, 
    default None.    
    '''
    try:
        f = float(s)
    except ValueError:
        return fail
    
    return f

def flatten(l, ltypes=(list, tuple)):
    """
    Flattens nested lists, algorithm courtesy of
    Mike C. Fletcher's BasicTypes library.
    """
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

def _reporthook(a,b,c): 
    # ',' at the end of the line is important!
    print "% 3.1f%% of %d bytes\r" % (min(100, float(a * b) / c * 100), c),
    sys.stdout.flush()
    
def download_file(url, dest_file):
    '''
    Downloads a file given by a url (such as one of the zipfiles
    for the kent ridge biomedical repository), and stores the
    file to the destination directory.
    @param url: The full url to the file to be downloaded. ifr.LUNG_CANCER_URL,
    ifr.PROSTATE_URL, and ifr.BCELL_LYMPHOMA_URL, are constants that point
    to the appropriate file.
    @param dest_dir: The destination filename, including full path.
    @return: (filename, headers)
    '''
    rc = urllib.urlretrieve(url, dest_file, reporthook=_reporthook)
    return rc

def sendEmailNotification(from_address, to_address_list, subject_str, body_str,
                          smtp_server='smtp.gmail.com', use_ssl=True,
                          login=None):
    '''
    This function makes it easy to send a short email message, for example, to
    notify people when a batch job has completed, or failed, etc.
    @param from_address: A string representing the e-mail address of the sender
    @param to_address_list: This is a LIST of strings representing the e-mail
    recipients. Or a single string if only one recipient.
    @param subject_str: A short string for the e-mail subject
    @param body_str: The text for the message body.
    @param use_ssl: Set to true if smtp connection requires ssl
    @param login: If the smtp server requires user authentication, set login=(username,password),
    else leave None.
    '''
    m = email.message_from_string(body_str)
    if type(to_address_list) == str:
        to_address_list = [to_address_list]
        
    m['To'] = ", ".join(to_address_list)
    m['From'] = from_address
    m['Subject'] = subject_str
    
    if use_ssl:
        s = smtplib.SMTP_SSL(smtp_server)
    else:
        s = smtplib.SMTP(smtp_server)
        
    if not login is None:
        try:
            s.login(*login)
        except(smtplib.SMTPAuthenticationError):
            print "================================================================="
            print "Error: Login to SMTP server failed. Invalid username or password."
            print "================================================================="
            s.quit()
            return
        
    rc = s.sendmail( from_address, to_address_list, m.as_string())
    if len(rc) > 0:
        print "Error(s) may have occurred during sending."
        print rc
        
    s.quit()

def plotFirst3PCA(X, labels=None, colors=None):
    '''
    Computes the first 3 principal components of the data
    matrix X, and shows the samples projected onto the 3 largest
    components using scatter3d()
    @param X: Input data, samples are in rows. It is advised to
    at least mean-center the data, but also to scale each input feature
    by dividing by standard deviation. Use svo_util.normalize() to
    do this.
    @param labels: A vector with length = rows(X), which has an integer
    label that indicates which class each sample belongs to. None means
    that the data is not classified, so all points will have the same
    color.
    @param colors: A list of color strings or numbers,
    one per label so that all points with the same label
    are colored the same. len(colors) == len( unique(labels) )
    @return: (T, W) where T is the data in pca-space and W are the
    loading weights. T and W can be used to reconstruct points from
    PCA space back to the 'normal' space, as with the function
    reconstructPCA().
    '''
    U,s,Vt = LA.svd(X, full_matrices=True)
    N,p = X.shape
    S = LA.diagsvd(s,N,p)
    T = U.dot(S)  #samples in PCA space (also, T = X.dot(V) where V=Vt.T)
    
    XYZ = T[:,0:3]  #first 3 columns are for the 3 largest components
    scatter3d(XYZ, labels=labels, colors=colors)
    
    return T, Vt.T  #return the transformed data, and the loading weights

def scatter3d(XYZ, labels=None, colors=None, cmap='gist_rainbow', lines=False, ax=None, show=True):
    '''
    Uses matplotlib and the Axes3D class to create
    a 3d scatter plot of data in matrix.
    @param XYZ: data matrix, rows are data points, and
    columns are the x,y, and z coordinates respectively
    @param labels: A vector of integer labels, same length
    as rows of XYZ, specifying the label or class to be
    assigned to each sample. Samples with same label are
    colored the same. If None, all samples will have the
    same label.
    @param colors: A list of color strings or numbers,
    one per label so that all points with the same label
    are colored the same. len(colors) == len( unique(labels) )
    @param cmap: The matplotlib colormap to use if colors=None
    @param lines: If true, then points are assumed sorted,
    and they will be connected with a line. False is default,
    and is just a scatter plot of the data.
    @param show: If True (default) the graph is displayed
    upon creation. Set to False if you want to do some post-processing
    on the graph before showing it.
    '''
    N, p = XYZ.shape
    assert p==3, "Input data must have 3 columns for 3D plotting!"
    
    if labels is None:
        labels = sp.array( [0]*N ) #vector length N with all zeros
    
    label_set = sp.unique(labels)
    
    if colors is None:
        cm = pl.get_cmap(cmap)
        num_colors = len(label_set)
        colors = [ cm(1.0*i/num_colors) for i in range(num_colors)]
        
    datasets = []  #break up XYZ into subsets, one per label
    for cur_label in label_set:
        cut = XYZ[ labels==cur_label, :]
        datasets.append(cut)
        
    if ax is None:
        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')
        
    for data, curr_color in zip(datasets,colors):
        if lines:
            ax.plot( data[:,0], data[:,1], data[:,2],
                 'o--', c=curr_color)
        else:
            ax.scatter( data[:,0], data[:,1], data[:,2],
                 'o', c=curr_color)
        
    pl.draw()
    if show: pl.show()

def scaleTo8bit(X):
    '''
    For array x, values are shifted to be all positive and then scaled according to
    the max value to lie in range 0-255.
    '''
    minx = X.min()
    Y = X - minx if minx >= 0 else X + minx
    maxx = X.max()
    Y /= maxx
    Y = Y * 255
    return Y

def reconstructPCA(T,W,components=None, means=None, stds=None):
    '''
    Reconstruct the data from PCA matrices T (the transformed data)
    and W (the loadings/weights). Can either reconstruct using all
    components or just the top N. If the data was mean centered
    and unit deviation normalized, providing the original data means
    and stds will complete the reconstruction.
    '''
    (_N,p) = W.shape
    if components is None:
        components = p 
        
    Tx = T[:,0:components]
    Wx = W[:,0:components]
    XnHat = Tx.dot(Wx.T)  #T = XW, W is orthog, therefore, X = TW'
    
    if not stds is None:
        Xhat = XnHat*stds+means
    else:
        Xhat = XnHat
        
    return Xhat
    


if __name__ == '__main__':
    pass
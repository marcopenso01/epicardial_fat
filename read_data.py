train_folder1 = r'G:\11\BURIC_MARIO_060Y(1)\series11002-unknown'

path = r'G:\11'
data_path1 = os.path.join(path, 'train1.hdf5')
hdf5_file1 = h5py.File(data_path1, "w")

addrs = []
fase = 11
slic_first = 7
slic_last = 15

frame = (30*(slic_first-1)+fase)-1
addrs.append(sorted(os.listdir(train_folder1))[frame])

for i in range(slic_last-slic_first):
    frame = frame+30
    addrs.append(sorted(os.listdir(train_folder1))[frame])

hdf5_file1.create_dataset('rows', (1,), dtype=np.uint16)
hdf5_file1.create_dataset('cols', (1,), dtype=np.uint16)
hdf5_file1.create_dataset('x', (1,), dtype=np.float32)
hdf5_file1.create_dataset('y', (1,), dtype=np.float32)
hdf5_file1.create_dataset('z', (1,), dtype=np.float32)
hdf5_file1.create_dataset('data', (len(addrs),512,512,3), dtype=np.uint16)

for i in range(len(addrs)):
    dcmPath = os.path.join(train_folder1, addrs[i])
    dicom_dataset = pydicom.dcmread(dcmPath)
    hdf5_file1['data'][i,...] = dicom_dataset.pixel_array
    if i == 0:
        hdf5_file1['rows'][i,...] = dicom_dataset.Rows
        hdf5_file1['cols'][i,...] = dicom_dataset.Columns
        hdf5_file1['x'][i,...] = 0.6641
        hdf5_file1['y'][i,...] = 0.6641
        hdf5_file1['z'][i,...] = abs(dicom_dataset.SliceThickness)
        
hdf5_file1.close()


file1 = r'G:\11\train1.hdf5'
fold1 = r'G:\11\img1'
data = h5py.File(file1,'r')
for i in range(len(data['data'])):
    img = np.array(data['data'][i]).astype("uint16")
    img = cv2.normalize(img, dst=None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX)
    img = img.astype("uint8")
    cv2.imwrite(os.path.join(fold1, str(i) + '.png'), img)
data.close()

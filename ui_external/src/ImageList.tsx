import {Image} from 'antd';
import React from 'react';

const mockImageList = [
    {
        id: 1,
        src: 'http://127.0.0.1:5000/picture/1',
        title: 'Image 1',
        description: 'Image 1 description',
        height: 200,
        width: 200,
    },
    {
        id: 2,
        src: 'https://gw.alipayobjects.com/zos/rmsportal/JiqGstEfoWAOHiTxclqi.png',
        title: 'Image 2',
        description: 'Image 2 description',
        height: 200,
        width: 200,
    },
    {
        id: 3,
        src: 'https://gw.alipayobjects.com/zos/rmsportal/JiqGstEfoWAOHiTxclqi.png',
        title: 'Image 3',
        description: 'Image 3 description',
        height: 200,
        width: 200,
    },
    {
        id: 4,
        src: 'https://gw.alipayobjects.com/zos/rmsportal/JiqGstEfoWAOHiTxclqi.png',
        title: 'Image 4',
        description: 'Image 4 description',
        height: 200,
        width: 200,
    }
];

const ImageList = () => {
    const handleDragEnd = (e) => {
        // set the image src data
        e.dataTransfer.setData('image', e.target.src);
        // also set the image height and width
        e.dataTransfer.setData('imageHeight', e.target.height);
        e.dataTransfer.setData('imageWidth', e.target.width);
    }
    return (
        <div className='flex flex-col'>
            {mockImageList.map((image) => {
                return (
                    <div className='m-2' key={image.id}>
                        <Image
                        key={image.id}
                        src={image.src}
                        height={image.height}
                        width={image.width}
                        onDragStart={handleDragEnd}
                        preview={false}
                        />
                    </div>
                )
            })}
        </div>
    )
}

export default ImageList;
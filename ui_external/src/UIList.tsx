import {Image} from 'antd';
import React,{useEffect} from 'react';
const BASIC_URL = "http://127.0.0.1:5000/";

// UI list shuold also contain the according JSON description, thus need a name
const mockUIList = [
    {
        id: 1,
        name: 'Android_1',
        height: 200,
        width: 200,
    },
    {
        id: 2,
        name: 'Android_2',
        title: 'Image 2',
        description: 'Image 2 description',
        height: 200,
        width: 200,
    },
    {
        id: 3,
        name: 'Android_3',
        title: 'Image 3',
        description: 'Image 3 description',
        height: 200,
        width: 200,
    }
];

const UIList = () => {
    const handleDragEnd = (e) => {
        //set the type
        // e.dataTransfer.type('type', 'ui');
        console.log(e.target,"e.target")
        e.dataTransfer.setData('filename', e.target.src);
        e.dataTransfer.setData('imageHeight', e.target.height);
        e.dataTransfer.setData('imageWidth', e.target.width);
    }

    useEffect(() => {
        const eventHandler = (e) => {
            console.log(e,"e")
            if(!e.data.pluginMessage) {
                return;
            }
            const {type=null, data} = e.data.pluginMessage;
            if (type !== 'uiSelectionChanged') {
                return;
            }
            const blob = new Blob([data], { type: 'image/png' });
            // create a FormData object to send the Blob data to the server
            const formData = new FormData();
            formData.append('image', blob);
            fetch('http://127.0.0.1:5000/api/image', {
                method: 'POST',
                body: formData
            }).then((response) => {
                console.log(response,"response")
            }).catch((error) => {
                console.log(error,"error")
            })
        }
        window.addEventListener('message', eventHandler);
        return () => {
            window.removeEventListener('message', eventHandler);
        }
    }, []);

    useEffect(() => {
        // get the real image data from the server

    },[])

    return (
        <div className='flex flex-col'>
            {mockUIList.map((image,index) => {
                return (
                    <div className='m-2' key={image.id} >
                        <Image
                        key={image.id}
                        src={BASIC_URL +"picture/"+image.name}
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

export default UIList;
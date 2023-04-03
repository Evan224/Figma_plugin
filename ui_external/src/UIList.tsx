import {Image} from 'antd';
import { Button } from 'antd';
import React,{useEffect,useState} from 'react';
const BASIC_URL = "http://127.0.0.1:5000/";
const DEFAULT_WIDTH = 144;
const DEFAULT_HEIGHT = 256;
import { useNavigate } from 'react-router-dom';
import Slider from "@madzadev/image-slider";

// UI list shuold also contain the according JSON description, thus need a name
const mockUIList = [
    {
        id: 1,
        name: 'Android_1',
        height: DEFAULT_HEIGHT,
        width: DEFAULT_WIDTH,
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
    const [uiList, setUIList] = useState(mockUIList);
    const navigate = useNavigate();
    const handleDragEnd = (e,name) => {
        e.dataTransfer.setData('name', name);
        e.dataTransfer.setData('filename', e.target.src);
        e.dataTransfer.setData('imageHeight', e.target.height*10);
        e.dataTransfer.setData('imageWidth', e.target.width*10);
    }

    const images = [
        { url: "https://picsum.photos/seed/a/1600/900" },
        { url: "https://picsum.photos/seed/b/1920/1080" },
        { url: "https://picsum.photos/seed/c/1366/768" },
      ];

    useEffect(() => {
        const eventHandler = (e) => {
            if(!e.data.pluginMessage) {
                return;
            }
            
            const {type=null, data} = e.data.pluginMessage;
            if (type == 'uiSelectionChanged') {
                const blob = new Blob([data], { type: 'image/png' });
                // create a FormData object to send the Blob data to the server
                const formData = new FormData();
                formData.append('image', blob);
                fetch('http://127.0.0.1:5000/api/image', {
                    method: 'POST',
                    body: formData
                }).then((response) => {
                    // console.log(response,"response")
                }).catch((error) => {
                    console.log(error,"error")
                })
            }

        }
        window.addEventListener('message', eventHandler);
        return () => {
            window.removeEventListener('message', eventHandler);
        }
    }, []);

    useEffect(() => {
        // get the real image data from the server
        fetch(BASIC_URL + "api/imageList").then((response) => response.json()).then((data)=>{
            const mappedData = data.imageList.map((item,index) => {
                return {
                    id: index,
                    name: item.slice(0,-4),
                    height: DEFAULT_HEIGHT,
                    width: DEFAULT_WIDTH,
                }
            })
            setUIList(mappedData)
        }).catch((error) => {
            console.log(error,"error")
        })

    },[])

    const imageList = uiList.map((image,index) => {
        return {url:BASIC_URL +"picture/"+image.name}
    })

    return (     
        <div className='flex flex-wrap w-[86vw] mx-auto justify-betweeen'>
            {uiList.map((image,index) => {
                return (
                    <div className='m-2 flex items-center' key={image.id} >
                        <Image
                        key={image.id}
                        src={BASIC_URL +"picture/"+image.name}
                        height={image.height}
                        width={image.width}
                        onDragStart={(e)=>handleDragEnd(e,image.name)}
                        preview={false}
                        className='cursor-pointer shadow-lg'
                        onDoubleClick={() => {
                            navigate('/ui/'+image.name)
                        }}
                        />
                        {/* <div className='flex'>
                            <Slider imageList={images} width={1000} height={300} />
                        </div> */}
                        {/* <Button onClick={() => {
                            navigate('/ui/'+image.name)
                        } }>choose</Button> */}
                    </div>
                )
            })}
        </div>
    )
}

export default UIList;
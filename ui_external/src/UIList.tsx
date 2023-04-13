import {Image} from 'antd';
import { Button } from 'antd';
import React,{useEffect,useState} from 'react';
const BASIC_URL = "http://127.0.0.1:5000/";
const DEFAULT_WIDTH = 144;
const DEFAULT_HEIGHT = 256;
import { useNavigate } from 'react-router-dom';
import Slider from "@madzadev/image-slider";

// UI list shuold also contain the according JSON description, thus need a name


const handleDoubleClickUI =async (event,ui) => {
    console.log(event.target,'event.target')
    // fetch the data /json/<string:pic_name>'
    const response = await fetch(BASIC_URL + 'json/' + ui);
    const data = await response.json();
    console.log(data,'data')
    const uiInfo={
        ui_name:ui,
        src:event.target.src,
        width:data.width,
        height:data.height,
    }
    parent.postMessage({ pluginMessage: { uiInfo,type:"setUIComponent" },pluginId:"1214978636007916809" }, '*');
  }

const UIList = () => {
    const [uiList, setUIList] = useState([]);
    const navigate = useNavigate();
    const handleDragEnd = (e,name) => {
        e.dataTransfer.setData('name', name);
        e.dataTransfer.setData('filename', e.target.src);
        e.dataTransfer.setData('imageHeight', e.target.height*10);
        e.dataTransfer.setData('imageWidth', e.target.width*10);
    }


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

    useEffect(() => {
        const eventHandler = (e) => {
            if(!e.data.pluginMessage) {
                return;
            }
            const {type=null, data} = e.data.pluginMessage;
            if (type == 'initializeMainPage') {
                // convert the id string to number
                if(data){
                    navigate('/ui/'+data)
                }
            }
        }
        window.addEventListener('message', eventHandler);
        return () => {
            window.removeEventListener('message', eventHandler);
        }
        }, []);

    useEffect(() => {
        parent.postMessage({ pluginMessage: {type:"initializeMainPage"},pluginId:"1214978636007916809" }, '*');
    },[]);

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
                        onDoubleClick={async (e) => {
                            await handleDoubleClickUI(e,image.name)
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
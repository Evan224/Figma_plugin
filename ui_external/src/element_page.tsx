import React, {useEffect,useState} from 'react';
import {Image} from 'antd';
const BASIC_URL = "http://127.0.0.1:5000/";
const DEFAULT_WIDTH = 144;
const DEFAULT_HEIGHT = 256;
import { useParams } from 'react-router-dom';
import { ArrowLeftOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { Spin } from 'antd';

const mockElementList = [
    {
        element_name:"pic1",
        id:1,
        height:100,
        width:100,
        left:200,
        top:200,
    },
    {
        element_name:"pic2",
        id:2,
        height:100,
        width:100,
        left:400,
        top:300,
    },
    {
        element_name:"pic3",
        id:3,
        height:100,
        width:100,
        left:400,
        top:400,
    },
    {
        element_name:"pic4",
        id:4,
        height:100,
        width:100,
        left:500,
        top:500,
    }
]

export default function ElementPage() {
  const {ui} = useParams();
  const [loading, setLoading] = useState(false);
  // only need the element list to change. The text is not changable, and the point is to modify the location.
  // fake element list for now
  const [fixedElementList, setFixedElementList] = useState(mockElementList);
  const [elementList, setElementList] = useState<any>([])
  const [fileredElementList, setFileredElementList] = useState<any>([])
  const navigate = useNavigate();
  const handleDragEnd = (e,name) => {
    const cursorX = e.clientX;
    const cursorY = e.clientY;
  
    const image = e.target;
    const imageRect = image.getBoundingClientRect();
    const imageX = imageRect.left;
    const imageY = imageRect.top;
  
    const relativeX = cursorX - imageX;
    const relativeY = cursorY - imageY;
    console.log(relativeX,relativeY,"relativeX,relativeY")
    e.dataTransfer.setData('name', name);
    e.dataTransfer.setData('filename', e.target.src);
    e.dataTransfer.setData('imageHeight', e.target.height*10);
    e.dataTransfer.setData('imageWidth', e.target.width*10);
    e.dataTransfer.setData('relativeX', relativeX*10);
    e.dataTransfer.setData('relativeY', relativeY*10);
}

  const handleDoubleClick = (event,id) => {
    const element = fixedElementList.find(item => item.id === id);
    // filter the element from the in the fixedElementList that is 
    if(!element) {
        return;
    }
    const elementInfo = {
        id: element.id,
        element_name: element.element_name,
        height: element.height,
        width: element.width,
        left: element.left,
        top: element.top,
        ui_name: ui,
        src:BASIC_URL +"element/"+ element.element_name,
    }

    // filter whihc id is not in the elementList
    setElementList(elementList=>[...elementList,element]);

    parent.postMessage({ pluginMessage: { elementInfo },type:"autoSetLocation",pluginId:"1214978636007916809" }, '*');
  };

  useEffect(() => {
    console.log('elementList',elementList,'------------------')
    const options = {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
            elementList:elementList,
            fixedElementList:fixedElementList
        })
      };
    setLoading(true);
    fetch(BASIC_URL +"recommend", options)
    .then(response => response.json())
    .then(data => {
        console.log(data);
        setFixedElementList(data);
    })
    .finally(() => {
        setLoading(false);
    })
    
  }, [elementList]);



  useEffect(() => {
    const eventHandler = (e) => {
        if(!e.data.pluginMessage) {
            return;
        }
        const {type=null, data} = e.data.pluginMessage;
        if (type == 'uiElementChanged') {
            // convert the id string to number
            const elementList = data.map(item => {
                item.id = parseInt(item.id);
                return item;
            })
            setElementList(elementList);
        }
    }
    window.addEventListener('message', eventHandler);
    return () => {
        window.removeEventListener('message', eventHandler);
    }
    }, []);

  useEffect(() => {
    // fetch the json file
    fetch(BASIC_URL +"json/"+ui)
    .then(response => response.json())
    .then(data => {
        //todo
    })
  }, []);


  

  return (
    <>
        <div>
      <ArrowLeftOutlined className='m-4' style={{ fontSize: '20px' }} onClick={()=> navigate(-1)} />
    </div>
    <div className='w-4/5 mx-auto flex flex-col justify-center items-center'>
        <Image
        src={BASIC_URL +"picture/"+ui}
        height={DEFAULT_HEIGHT}
        width={DEFAULT_WIDTH}
        preview={false}
        onDragStart={(e)=>handleDragEnd(e,ui)}
        className='shadow-lg my-2'
        />
        {loading && <Spin spinning={loading} className="my-10"/>}
        
        {!loading && <div className='flex my-5 w-[95vw] flex-wrap'>
            {fixedElementList.map((item,index) => {
                // if item.id is found in the elementList, then we don't need to render the element
                const isFound = elementList.find(element => element.id === item.id);
                if(isFound) {
                    return null;
                }
                const height=200;
                const width=height*item.width/item.height;
                return (
                    
                    <div className='w-1/2 flex justify-center' key={item.id} >
                        <Image
                            key={item.id}
                            src={BASIC_URL +"element/"+item.element_name}
                            height={height}
                            width={width}
                            preview={false}
                            onDoubleClick={(e)=>{handleDoubleClick(e,item.id)}}
                            id={item.element_name}
                        />
                    </div>
                )
            })}
        </div>}

        
    </div>
    </>
  );
}
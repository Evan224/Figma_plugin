import React, {useEffect,useState} from 'react';
import {Image} from 'antd';
const BASIC_URL = "http://127.0.0.1:5000/";
const DEFAULT_WIDTH = 144; 
const DEFAULT_HEIGHT = 256;
import { useParams } from 'react-router-dom';
import { ArrowLeftOutlined } from '@ant-design/icons';
import { useNavigate,useLocation } from 'react-router-dom';
import { Spin } from 'antd';
import { Button, Modal } from 'antd';

// const mockElementList = [
//     {
//         element_name:"pic1",
//         id:1,
//         height:100,
//         width:100,
//         left:200,
//         top:200,
//     },
//     {
//         element_name:"pic2",
//         id:2,
//         height:100,
//         width:100,
//         left:400,
//         top:300,
//     },
//     {
//         element_name:"pic3",
//         id:3,
//         height:100,
//         width:100,
//         left:400,
//         top:400,
//     },
//     {
//         element_name:"pic4",
//         id:4,
//         height:100,
//         width:100,
//         left:500,
//         top:500,
//     }
// ]

export default function ElementPage() {
  const params = useParams();
  const [ui, setUI] = useState<any>(params.ui);
  const [loading, setLoading] = useState(false);
  // only need the element list to change. The text is not changable, and the point is to modify the location.
  // fake element list for now
  const [fixedElementList, setFixedElementList] = useState([]);
  const [target, setTarget] = useState<any>(null);
  const [elementList, setElementList] = useState<any>([])
  const [initialElementList, setInitialElementList] = useState<any>([])
  const navigate = useNavigate();
  const location = useLocation();
  const handleDragEnd = (e,name) => {
    const cursorX = e.clientX;
    const cursorY = e.clientY;
  
    const image = e.target;
    const imageRect = image.getBoundingClientRect();
    const imageX = imageRect.left;
    const imageY = imageRect.top;
  
    const relativeX = cursorX - imageX;
    const relativeY = cursorY - imageY;
    e.dataTransfer.setData('name', name);
    e.dataTransfer.setData('filename', e.target.src);
    e.dataTransfer.setData('imageHeight', e.target.height*5);
    e.dataTransfer.setData('imageWidth', e.target.width*5);
    e.dataTransfer.setData('relativeX', relativeX*5);
    e.dataTransfer.setData('relativeY', relativeY*5);

}

  const handleDoubleClick = (event,id) => {
    // get the element from the fixedElementList
    // if id is target, then make the target the element
    let element;
    if(id===target.id) {
        element = target;
    }else{
        element = fixedElementList.find(item => item.id === id);
    }

    // filter the element from the in the fixedElementList that is 
    if(!element) {
        return;
    }

    console.log(element,'element')
    const elementInfo = {
        id: element.id,
        element_name: element.id,
        height: element.height,
        width: element.width,
        left: element.left,
        top: element.top,
        ui_name: ui,
        src:BASIC_URL +"element/"+ ui+"/"+element.id,
    }
    // filter whihc id is not in the elementList
    const newList=[...elementList,element]
    setElementList(newList);
    setLoading(true);
    const options = {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            elementList:newList,
            ui:ui,
            fixedElementList:fixedElementList,
            element:element,
        })
    };
    fetch(BASIC_URL +"recommend", options).then(response => response.json()).then(data => {
        console.log(data,'data')
        parent.postMessage({ pluginMessage: { elementInfo },type:"autoSetLocation",pluginId:"1214978636007916809" }, '*');
    }).finally(() => { 
        setLoading(false);
    });


  };

  const setUpUIComponent =async () => {
    const response = await fetch(BASIC_URL + 'json/' + ui);
    const data = await response.json();
    const uiInfo={
        ui_name:ui,
        src:BASIC_URL +"picture/"+ui,
        width:data.width,
        height:data.height,
    }
    parent.postMessage({ pluginMessage: { uiInfo,type:"setUIComponent" },pluginId:"1214978636007916809" }, '*');
  }


  useEffect(() => {
    // fetch the json file using async/await
    const fetchData = async () => {
      try {
        const response = await fetch(BASIC_URL + 'json/' + ui);
        const data = await response.json();
        console.log('data',data,'-----------')
        if (data?.elements) {
          setFixedElementList(data.elements);
        }

        const response2= await fetch(BASIC_URL + 'partialjson/' + ui);
        const data2 = await response2.json();
        if (data2?.elements) {
            console.log('data2',data2)
            setInitialElementList(data2.elements);
            setTarget(data2.target);
        };
        await setUpUIComponent();

      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []);

//   useEffect(() => {
//     console.log(initialElementList,'initialElementList',elementList,'elementList')
//     initialElementList.forEach(element => {
//         const elementInfo = {
//             id: element.id,
//             element_name: element.id,
//             height: element.height,
//             width: element.width,
//             left: element.left,
//             top: element.top,
//             ui_name: ui,
//             src:BASIC_URL +"element/"+ ui+"/"+element.id,
//         };
//         const newList=[...elementList,element]
//         setElementList(newList);
//         parent.postMessage({ pluginMessage: { elementInfo,type:"initialSetUp" },pluginId:"1214978636007916809" }, '*');
//     });
//   }, [initialElementList]);

  const setUpInitElement = () => {
    initialElementList.forEach(element => {
        const elementInfo = {
            id: element.id,
            element_name: element.id,
            height: element.height,
            width: element.width,
            left: element.left,
            top: element.top,
            ui_name: ui,
            src:BASIC_URL +"element/"+ ui+"/"+element.id,
        };
        const newList=[...elementList,element]
        setElementList(newList);
        parent.postMessage({ pluginMessage: { elementInfo,type:"initialSetUp" },pluginId:"1214978636007916809" }, '*');
    });
  }

  useEffect(() => {
    const eventHandler = (e) => {
        if(!e.data.pluginMessage) {
            return;
        }
        const {type=null, data} = e.data.pluginMessage;
        if (type == 'uiElementChanged') {
            // convert the id string to number
            let elementList=[];
            data.map(item => {
                item.id = parseInt(item.id);
                if(item.parent_name === ui){
                    elementList.push(item);
                }
            })
            setElementList(elementList);
        }

        if (type == 'uiFinished') {
            console.log('uiFinished',data);
            console.log('uiFinished',initialElementList);
            setUpInitElement();
        }
    }
    window.addEventListener('message', eventHandler);
    return () => {
        window.removeEventListener('message', eventHandler);
    }
    }, [initialElementList]);

  useEffect(() => {
    parent.postMessage({ pluginMessage: {type:"initializeList"},pluginId:"1214978636007916809" }, '*');
  },[]);


  if(!target) {
    return null;
  }

  let {height,width}=target;
  const RATIO=height/width;
  if(height>width){
      height=100;
      width=height/RATIO;
  }else{
      width=100;
      height=width*RATIO;
  }
  


  return (
    <>
        <div className='flex items-center'>
      <ArrowLeftOutlined className='m-4' style={{ fontSize: '20px' }} onClick={()=> navigate(-1)} />
      All the UIs
    </div>
    <div className='w-4/5 mx-auto flex flex-col justify-center items-center'>
        <Image
        src={BASIC_URL +"picture/"+ui}
        height={DEFAULT_HEIGHT}
        width={DEFAULT_WIDTH}
        preview={false}
        // onDragStart={(e)=>handleDragEnd(e,ui)}
        // onDoubleClick={(e)=>{handleDoubleClickUI(e)}}
        className='shadow-lg my-2'
        />
        {loading && (
        <>
            <div className="fixed inset-0 bg-black bg-opacity-50 z-10"></div>
            <Spin
            spinning={loading}
            size="large"
            className="fixed z-20 left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2"
            />
        </>
        )}

        <div className='w-full my-5 flex justify-between'>
            <div className='flex-col flex'>
                <div>Recommend Next</div>
                <Image
                    key={target.id}
                    src={BASIC_URL +"element/"+ui+"/"+target.id}
                    height={height}
                    width={width}
                    preview={false}
                    onDoubleClick={(e)=>{handleDoubleClick(e,target.id)}}
                    id={target.id}
                    className='shadow-lg rounded-lg my-5'
                />
            </div>
            <div className='flex-col flex'>
                {/* <Button type="text" onClick={()=>{}}>init</Button> */}
                <Button type="text" onClick={()=>{
                    setElementList([]);
                    parent.postMessage({ pluginMessage: {type:"reset",ui:ui,id:target.id},pluginId:"1214978636007916809" }, '*');
                }}>reset</Button>
            </div>
        </div>


        
        <div className='flex my-5 w-[95vw] flex-wrap'>
            {initialElementList.map((item,index) => {
                // if item.id is found in the elementList, then we don't need to render the element
                const isFound = elementList.find(element => element.id === item.id);
                const ifTaget=target.id===item.id;
                if(isFound||ifTaget) {
                    return null;
                }
                let {height,width}=item;
                const RATIO=height/width;
                if(height>width){
                    height=120;
                    width=height/RATIO;
                }else{
                    width=120;
                    height=width*RATIO;
                }

                return (
                    <div className='w-1/2 flex justify-center h-[150px] items-center' key={item.id} >
                        <div className='w-full flex justify-center items-center'>
                        <Image
                            className='shadow-lg rounded-lg'
                            key={item.id}
                            src={BASIC_URL +"element/"+ui+"/"+item.id}
                            height={height}
                            width={width}
                            preview={false}
                            onDoubleClick={(e)=>{handleDoubleClick(e,item.id)}}
                            id={item.id}
                        />
                        </div>
                    </div>
                )
            })}
        </div>

        
    </div>
    </>
  );
}
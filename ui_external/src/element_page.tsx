import React, {useEffect,useState} from 'react';
import {Image} from 'antd';
const BASIC_URL = "http://127.0.0.1:5000/";
const DEFAULT_WIDTH = 144; 
const DEFAULT_HEIGHT = 256;
import { useParams} from 'react-router-dom';
import { ArrowLeftOutlined } from '@ant-design/icons';
import { useNavigate,useLocation } from 'react-router-dom';
import { Spin } from 'antd';
import { Button, Modal } from 'antd';
import { RightCircleOutlined,LeftCircleOutlined} from '@ant-design/icons';
import { Divider, Space, Tag,Badge } from 'antd';
import UIList from './UIList';
import {debounce, throttle} from './utility';

const handleDoubleClickUI =async (event,ui) => {

    const response = await fetch(BASIC_URL + 'json/' + ui);
    const data = await response.json();
    const uiInfo={
        ui_name:ui,
        src:event.target.src,
        width:data.width,
        height:data.height,
    }
    parent.postMessage({ pluginMessage: { uiInfo,type:"setUIComponent" },pluginId:"1214978636007916809" }, '*');
}

export default function ElementPage() {
    const [uiList, setUIList] = useState([]);
    const [hoveredElement, setHoveredElement] = useState<any>({
        width: 200,
        height: 200,
        top:200,
        left: 200
    });
    const params = useParams();
    const [ui, setUI] = useState<any>(params.ui);
    const [loading, setLoading] = useState(false);
    const [fixedElementList, setFixedElementList] = useState([]);
    const [libraryElementList, setLibraryElementList] = useState([]);
    const [target, setTarget] = useState<any>(null);
    const [elementList, setElementList] = useState<any>([])
    const [initialElementList, setInitialElementList] = useState<any>([])
    const [restElementList, setRestElementList] = useState<any>([])
    const navigate = useNavigate();

    const [xScale, setXScale] = useState(1);
    const [yScale, setYScale] = useState(1);

    const [thumbnail, setThumbnail] = useState<any>(null);
    const [helper, setHelper] = useState(true);


    useEffect(() => {

        const fetchData=async()=>{
            //post
            const options = {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    name:ui,
                    element_list:elementList
                }),
            };

            // /api/element_image
            const response = await fetch(BASIC_URL + 'api/element_image', options);
            // const response = await fetch(BASIC_URL + 'json/' + ui);
            const data = await response.json();
            setThumbnail(data?.image_url);
        }
        fetchData();
    }, [ui,elementList]);


    useEffect(() => {
        // get the elements from the fixedElementList which is not in the elementList
        const newElementList = fixedElementList.filter(item => {
            return !elementList.find(element => element.id === item.id);
        })
        setRestElementList(newElementList);
    },[fixedElementList,elementList]);

    useEffect(() => {
        console.log('fixedElementList',fixedElementList,'----------------')
        console.log('ElementList',elementList,'----------------')
        console.log('restElementList',restElementList,'----------------')
    },[restElementList]);

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

    const handleDoubleClick =async (event,id) => {
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
        setLoading(true);

        const response = await fetch(BASIC_URL + 'json/' + ui);
        const all_json = await response.json();
        all_json.elements = elementList;
        all_json.target=element;

        const options = {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                data:all_json
            })
        };
        const elementInfo = {
            id: element.id,
            element_name: element.id,
            height: element.height,
            width: element.width,
            left: helper ? element.left : `calc(50% - ${element.width / 2}px)`, // Set to center if helper is false
            top: helper ? element.top : `calc(50% - ${element.height / 2}px)`, // Set to center if helper is false// Set to center if helper is false
            ui_name: ui,
            src:BASIC_URL +"element/"+ ui+"/"+element.id,
        }
        // filter whihc id is not in the elementList
        const newList=[...elementList,element]
        setElementList(newList);
        parent.postMessage({ pluginMessage: { elementInfo },type:"autoSetLocation",pluginId:"1214978636007916809" }, '*');
        setLoading(false);
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

    const updateElementList = async() => {
        // get the current element list
        const response = await fetch(BASIC_URL + 'json/' + ui);
        const data = await response.json();
        
        data.elements=elementList;
        const options = {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                data:data,
                name:ui
            }),
        }
        const response2=await fetch(BASIC_URL + 'recommend/mock', options);
        const data2 = await response2.json();

        if(data2?.target){
            setFixedElementList([...elementList,...data2.target]);
        }
    }

    // todo: decide on the time to update the dotted rectangle
    // const updateDottedLine = async(elementList) => {

    //     const options = {
    //         method: "POST",
    //         headers: {
    //             "Content-Type": "application/json"
    //         },
    //         body: JSON.stringify({
    //             data:elementList,
    //             name:ui
    //         }),
    //     }
    //     const response=await fetch(BASIC_URL + 'dotted/mock', options);
    //     const data = await response.json();

    //     parent.postMessage({ pluginMessage: { data:{...data,ui},type:"updateDottedLine" },pluginId:"1214978636007916809" }, '*');
    // }

    useEffect(() => {
        // updateElementList();
    },[elementList])

    useEffect(() => {
        // fetch the json file using async/await
        setElementList([]);
        const fetchData = async () => {
            try {
                const response = await fetch(BASIC_URL + 'json/' + ui);
                const data = await response.json();
                setXScale(data.width/DEFAULT_WIDTH);
                setYScale(data.height/DEFAULT_HEIGHT);

                if (data?.elements) {
                    setFixedElementList(data.elements);
                }

                const response2= await fetch(BASIC_URL + 'partialjson/' + ui);
                const data2 = await response2.json();
                if (data2?.elements) {
                    setInitialElementList(data2.elements);
                    setTarget(data2.target);
                };
                await setUpUIComponent();

                // const response3= await fetch(BASIC_URL + 'recommend/mock');

            } catch (error) {
                console.error('Error fetching data:', error);
            }
        };

        fetchData();
    }, [ui]);

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
            function areArraysEqual(arr1, arr2) {
                if (arr1.length !== arr2.length) {
                  return false;
                }
              
                for (let i = 0; i < arr1.length; i++) {
                  if (JSON.stringify(arr1[i]) !== JSON.stringify(arr2[i])) {
                    return false;
                  }
                }
              
                return true;
              }
            let newElementList=[];
            data.map(item => {
                item.id = parseInt(item.id);
                if(item.parent_name === ui){
                    newElementList.push(item);
                }
            });
            setElementList(newElementList);
        }

        if (type == 'uiFinished') {
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
        <ArrowLeftOutlined className='m-4' style={{ fontSize: '20px' }} onClick={()=>  navigate('/')} />
        All the UIs
        </div>
        <div className='w-full mx-auto flex flex-col justify-center items-center'>
            <div className='flex justify-between items-center w-full sticky top-0 z-30 bg-white'>
                <LeftCircleOutlined style={{ fontSize: '20px', marginLeft:"10px"}} onClick={
                    ()=> {
                        // also consider the -1 case
                        parent.postMessage({ pluginMessage: {type:"cleanUp" },pluginId:"1214978636007916809" }, '*');
                        const index = uiList.findIndex(item => item.name === ui);
                        let previousUI;
                        if(index === 0){
                            previousUI = uiList[uiList.length-1].name;
                        }else{
                            previousUI = uiList[index - 1].name;
                        }
                        navigate(`/ui/${(previousUI)}`);
                        setUI(previousUI);
                    }
                }/>
                <div
                    style={{ position: 'relative', display: 'inline-block' }} 
                    className='shadow-lg my-2'
                >
                    <Image
                        src={thumbnail}
                        height={DEFAULT_HEIGHT}
                        width={DEFAULT_WIDTH}
                        preview={false}
                        onDoubleClick={(e)=>{handleDoubleClickUI(e,ui)}}
                    />
                    {helper &&hoveredElement.width && (
                        <div 
                            style={{
                                position: 'absolute',
                                top: `${hoveredElement.top}%`,
                                left: `${hoveredElement.left}%`,
                                width: `${hoveredElement.width}%`,
                                height: `${hoveredElement.height}%`,
                                boxShadow: '0 0 10px rgba(0, 0, 0, 1)', // Add shadow effect
                                backgroundColor: 'rgba(0, 0, 0, 0.9)', // Fill the box with semi-transparent black
                                zIndex: 10
                            }}
                        >
                            {/* {"this is the place where the element will be placed"} */}
                        </div>
                    )}
                </div>
                        
                <RightCircleOutlined style={{ fontSize: '20px',marginRight:"10px"}} onClick={
                    ()=> {
                        parent.postMessage({ pluginMessage: {type:"cleanUp" },pluginId:"1214978636007916809" }, '*');
                            const index = uiList.findIndex(item => item.name === ui);
                            let nextUI;
                            if(index === uiList.length-1){
                                nextUI = uiList[0].name;
                            }else{
                                nextUI = uiList[index + 1].name;
                            }
                            navigate(`/ui/${(nextUI)}`);
                            setUI(nextUI);
                        }
                }/>
            </div>
            <div className='flex w-4/5 flex-col sticky top-0'>
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
                        <Button type="text" onClick={()=>{
                            parent.postMessage({ pluginMessage: {type:"cleanUp" },pluginId:"1214978636007916809" }, '*');
                            window.location.reload();
                        }}>reset</Button>
                        <Button type="text" onClick={() => setHelper(prevHelper => !prevHelper)}>
                        Toggle Helper
                        </Button>
                    </div>
                </div>
            </div>
            {/* <div className=''> <Tag color="red">high</Tag> </div> */}
            
            <div className='flex my-5 w-[95vw] flex-wrap'>
            {restElementList.map((item,index) => {
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
                <div className='w-1/2 flex justify-center h-[150px] items-center shadow-sm hover:shadow-2xl' key={item.id}
                    onMouseEnter={() => {
                        setHoveredElement({ 
                            width: (item.width / xScale /1.44).toFixed(1), 
                            height: (item.height / yScale / 2.56).toFixed(1),
                            top:(item.top / yScale / 2.56).toFixed(1),
                            left:(item.left / xScale / 1.44).toFixed(1)
                        })
                    }}
                    onMouseLeave={() => setHoveredElement({})}
                >
                    {helper ? (
                    <Badge.Ribbon text={item.level||'high'} color={item.level==="high"?"red":"blue"}>
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
                    </Badge.Ribbon>
                    ):                        <div className='w-full flex justify-center items-center'>
                    <Image
                        className='shadow-lg rounded-lg'
                        key={item.id}
                        src={BASIC_URL +"element/"+ui+"/"+item.id}
                        height={height}
                        width={width}
                        preview={false}
                        onDoubleClick={(e)=>{handleDoubleClick(e,item.id)}}
                        id={item.id}
                        onDragEnd={(e)=>{handleDragEnd(e,item.id)}}
                    />
                </div>}:
                </div>

                )
            })}
            </div>

        </div>
    </>
  );
}
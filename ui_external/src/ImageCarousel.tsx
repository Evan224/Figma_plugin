import React from 'react';
import { Carousel } from 'antd';
import {Image} from 'antd';
const BASIC_URL = "http://127.0.0.1:5000/";
import Slider from "@madzadev/image-slider";
import "@madzadev/image-slider/dist/index.css";

const contentStyle: React.CSSProperties = {
  margin: 0,
  height: '160px',
  color: '#fff',
  lineHeight: '160px',
  textAlign: 'center',
  background: '#364d79',
};

const App: React.FC = ({testImage}) => {
  const onChange = (currentSlide: number) => {
    console.log(currentSlide);
  };

  console.log(testImage,"testImage")

  return (
    <Carousel afterChange={onChange}>  
        <div className='flex flex-wrap w-[86vw] mx-auto justify-betweeen'>
            {testImage.map((image,index) => {
                return (
                    <div key={image.id} >
                        <Image
                        key={image.id}
                        src={BASIC_URL +"picture/"+image.name}
                        height={image.height}
                        width={image.width}
                        preview={false}
                        className='cursor-pointer shadow-lg'
                        />
                    </div>
                )
            })}
        </div>
    </Carousel>
  );
};

export default App;
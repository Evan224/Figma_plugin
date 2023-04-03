
// import "slick-carousel/slick/slick.css"; 
// import "slick-carousel/slick/slick-theme.css";
// import {Image} from 'antd';

// import React, { Component } from "react";
// import Slider from "react-slick";

// export default class CenterMode extends Component {
//   render() {
//     const settings = {
//       className: "center",
//       centerMode: true,
//       infinite: true,
//       centerPadding: "60px",
//       slidesToShow: 3,
//       speed: 500
//     };
//     const { imageList, BASIC_URL } = this.props;
//     return (
//       <div>
//         <h2>Center Mode</h2>
//         <Slider {...settings}>
//         {imageList.map((image:any) => {
//             console.log(BASIC_URL +"picture/"+image.name,"image.name")
//             return (
//                 <div className="w-full">
//                 <Image
//                     key={image.id}
//                     src={BASIC_URL +"picture/"+image.name}
//                     height={image.height}
//                     width={image.width}
//                     preview={false}
//                     className='cursor-pointer shadow-lg'
//                     />
//                 </div>
//             )
//         })
//         }
//         </Slider>
//       </div>
//     );
//   }
// }



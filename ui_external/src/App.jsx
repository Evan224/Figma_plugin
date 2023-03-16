import React from 'react'
import ImageList from './ImageList'

export default function App() {
  return (
    <div className='flex flex-col justify-center items-center'>
      <h1 className="text-3xl font-bold">
        Generated Images
      </h1>
      <ImageList />
    </div>
  )
}
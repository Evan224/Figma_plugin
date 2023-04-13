import React from 'react'
import UIList from './UIList'


export default function App() {
  return (
    <div className='flex flex-col justify-center items-center'>
      <h1 className="text-3xl font-bold w-2/3 flex justify-center my-4">
        Choose a UI 
      </h1>
      <UIList />
    </div>
  )
}
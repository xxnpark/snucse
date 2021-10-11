import "./Input.css"
import {useState} from "react";

const Input = ({setSearch}) => {
  const [searchContent, setSearchContent] = useState("")

  const handleSearch = (e) => {
    setSearchContent(e.target.value)
    setSearch(e.target.value)
  }

  return <input className="Input" id="search" value={searchContent} onChange={handleSearch}/>
}

export default Input;
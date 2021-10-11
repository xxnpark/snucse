import './App.css';
import Header from './components/Header/Header.js'
import Dashboard from "./components/Dashboard/Dashboard.js";
import Input from "./components/Input/Input.js";
import Button from "./components/Button/Button.js";
import Adder from "./components/Adder/Adder.js";
import StudentList from "./components/StudentList/StudentList.js";
import StudentInfo from "./components/StudentInfo/StudentInfo.js";
import {useState} from "react";
import dummyData from "./assignment-dummy-data.js"

function App() {
  const studentList = useState(dummyData);
  const selected = useState(0)
  const search = useState("");
  const adderSelected = useState("closed")

  const name = useState("");
  const grade = useState("");
  const profileImg = useState("");

  return (
    <div className="App">
      <Header/>
      <Dashboard/>
      <div className="Students">
        <div className="ManageStudentList">
          <div className="StudentListHeader">
            <Input setSearch={search[1]}/>
            <Button setAdderSelected={adderSelected[1]}/>
          </div>
          <StudentList search={search[0]} studentList={studentList} selected={selected} setName={name[1]} setGrade={grade[1]} setProfileImg={profileImg[1]}/>
          <Adder adderSelected={adderSelected} studentList={studentList} setSelected={selected[1]} setName={name[1]} setGrade={grade[1]} setProfileImg={profileImg[1]}/>
        </div>
        <div className="Divider"></div>
        <StudentInfo studentList={studentList} selected={selected} name={name} grade={grade} profileImg={profileImg}/>
      </div>
    </div>
  );
}

export default App;

import "./Adder.css"
import {useState} from "react";

const Adder = ({adderSelected, studentList, setSelected, setName, setGrade, setProfileImg}) => {
  const [addName, setAddName] = useState("");
  const [addGrade, setAddGrade] = useState("");
  const [addProfile, setAddProfile] = useState("");

  const kor = /^[ㄱ-ㅎ|가-힣]+$/;

  const addStudent = () => {
    if (kor.test(addName) && addName.length >= 2 && addName.length <= 3 && ["1", "2", "3"].includes(addGrade) && !studentList[0].map(item => item.name).includes(addName)) {
      studentList[1]([...studentList[0], {
        id: studentList[0].length + 1,
        name: addName,
        grade: addGrade,
        profileImg: addProfile,
      }]);
      setName(addName);
      setGrade(addGrade);
      setProfileImg(addProfile);
      setAddName("");
      setAddGrade("");
      setAddProfile("");
      adderSelected[1]("closed")
      setSelected(studentList[0].length + 1);
    } else {
      window.alert("학년과 이름을 올바르게 입력해주세요.");
    }
  }

  const onEnter = (e) => {
    if (e.key === "Enter") {
      addStudent();
    }
  }

  const closeAdder = () => {
    setAddName("");
    setAddGrade("");
    setAddProfile("");
    adderSelected[1]("closed")
  }

  return (
    <div className={`Adder ${adderSelected[0] === "open" && "open"}`}>
      <div className="Wrapper">
        <div className="Form">
          <div className="FormType">이름</div>
          <input className="FormInput" value={addName} onChange={(e) => setAddName(e.target.value)} onKeyPress={onEnter}/>
        </div>
        <div className="Form">
          <div className="FormType">학년</div>
          <input className="FormInput" value={addGrade} onChange={(e) => setAddGrade(e.target.value)} onKeyPress={onEnter}/>
        </div>
        <div className="Form">
          <div className="FormType">프로필</div>
          <input className="FormInput" value={addProfile} onChange={(e) => setAddProfile(e.target.value)} onKeyPress={onEnter}/>
        </div>
        <button className="Close" onClick={closeAdder}>닫기</button>
        <button className="Add" onClick={addStudent}>추가</button>
      </div>
    </div>
  );
}

export default Adder;
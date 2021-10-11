import "./StudentList.css"
import StudentItem from "./StudentItem/StudentItem.js";

const StudentList = ({search, studentList, selected, setName, setGrade, setProfileImg}) => {
  const searchList = () => {
    const tempList = [];
    for (const student of studentList[0]) {
      if (student.name.includes(search)) {
        tempList.push(student);
      }
    }
    return tempList;
  }

  return (
    <div className="StudentList">
      <div className="Type">
        <div className="Name">이름</div>
        <div className="Grade">학년</div>
      </div>
      <ul className="List">
        {searchList().map(item => (
          <StudentItem key={item.id} item={item} selected={selected} setName={setName} setGrade={setGrade} setProfileImg={setProfileImg}/>
        ))}
      </ul>
    </div>
  );
}

export default StudentList;
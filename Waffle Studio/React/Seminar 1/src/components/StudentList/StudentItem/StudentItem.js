import "./StudentItem.css"
import rightarrow from "./rightarrow.svg"

const StudentItem = ({item, selected, setName, setGrade, setProfileImg}) => {
  const select = () => {
    if (selected[0] !== item.id) {
      selected[1](item.id);
      setName(item.name);
      setGrade(item.grade);
      setProfileImg(item.profileImg);
    }
    else selected[1](0);
  }

  return (
    <li className={`ListElement ${selected[0] === item.id ? "Selected" : "NotSelected"}`}>
      <div className="NameElement">{item.name}</div>
      <div className="GradeElement">{item.grade}</div>
      <button className="Button" onClick={select}><img src={rightarrow} alt="button" className={`ButtonImg ${selected[0] === item.id ? "Selected" : "NotSelected"}`}/></button>
    </li>
  );
}

export default StudentItem
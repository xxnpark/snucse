import "./StudentInfo.css"

const StudentInfo = ({studentList, selected, name, grade, profileImg}) => {
  const kor = /^[ㄱ-ㅎ|가-힣]+$/;

  const saveChanges = () => {
    if (kor.test(name[0]) && name[0].length >= 2 && name[0].length <= 3 && [1, 2, 3].includes(grade[0])) {
      const tempStudentList = studentList[0].slice();
      tempStudentList.splice(selected[0] - 1, 1, {
        id: tempStudentList[selected[0] - 1].id,
        name: name[0],
        grade: grade[0],
        profileImg: profileImg[0],
      });
      studentList[1](tempStudentList);
    } else {
      window.alert("학년과 이름을 올바르게 입력해주세요.");
    }
  }

  const deleteStudent = () => {
    const tempStudentList = studentList[0].slice();
    tempStudentList.splice(selected[0] - 1, 1,);
    for (let i = selected[0] - 1; i < tempStudentList.length; i++) {
      tempStudentList[i].id--;
    }
    studentList[1](tempStudentList);
    selected[1](0);
  }

  return (
    <div className="StudentInfo">
      {selected[0] === 0
        ? <div className="Notice">왼쪽 표에서 학생을 선택해주세요</div>
        : <div>
          <button className="Save" onClick={saveChanges}>저장</button>
          <button className="Delete" onClick={deleteStudent}>삭제</button>
          <div className="profile" style={{backgroundImage: "url(" + profileImg[0] + ")"}}></div>
          <div className="Edit">
            <div className="EditType">이름</div>
            <input className="EditInput" id="name" value={name[0]}  onChange={(e) => name[1](e.target.value)}/>
          </div>
          <div className="Edit">
            <div className="EditType">학년</div>
            <input className="EditInput" id="grade" value={grade[0]} onChange={(e) => grade[1](e.target.value)}/>
          </div>
          <div className="Edit">
            <div className="EditType">프로필</div>
            <input className="EditInput" id="profileImg" value={profileImg[0]} onChange={(e) => profileImg[1](e.target.value)}/>
          </div>
        </div>}
    </div>
  );
}

export default StudentInfo;